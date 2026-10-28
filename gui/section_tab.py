"""
gui/section_tab.py

Tab 4: Heatmap Sections, with two sub-tabs:
  A. Frequency Slice - pick one microwave frequency, plot Signal vs Field
     for every loaded (processed) dataset overlaid, using user labels.
  B. Field Slice - pick one magnetic field value, plot Signal vs Frequency
     for every loaded (processed) dataset overlaid.

Both sub-tabs let the user export the currently displayed slice.
"""

from __future__ import annotations

from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QTabWidget,
    QVBoxLayout, QWidget,
)

# Matplotlib's default colour cycle ("tab10"). Plotted datasets take these
# in the order they were added to the Plotted column.
DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

import numpy as np

from gui.app_state import rescale_field_spin
from plotting.mpl_canvas import MplCanvas
from utils.exporters import export_slices_xy


class _BaseSliceWidget(QWidget):
    """Shared scaffolding for the Frequency-Slice and Field-Slice sub-tabs:
    two dataset columns (Available -> Plotted, plotted in the order added,
    coloured by the default colour cycle), a value spinbox, Extract + Export
    buttons, and a plot canvas. Subclasses implement the actual slicing
    and axis labelling.
    """

    x_label = "x"
    value_label = "Value"
    value_suffix = ""
    value_range = (-1e6, 1e6)
    value_default = 0.0
    value_decimals = 4
    x_range_kind = None  # "field" or "frequency": which global range limits the x-axis

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._last_x = None
        self._last_series = {}
        self._build_ui()
        self.app_state.processed_changed.connect(self._refresh_dataset_list)
        self.app_state.display_range_changed.connect(self._apply_ranges)
        self.app_state.field_unit_changed.connect(self._on_field_unit_changed)

    def _build_ui(self):
        root = QHBoxLayout(self)

        controls = QVBoxLayout()

        group = QGroupBox("Datasets")
        group_layout = QHBoxLayout(group)

        # Left column: every processed dataset not yet added to the plot.
        avail_col = QVBoxLayout()
        avail_col.addWidget(QLabel("Available"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.available_list.itemDoubleClicked.connect(lambda _i: self._add_selected())
        avail_col.addWidget(self.available_list)

        # Middle: move buttons.
        btn_col = QVBoxLayout()
        btn_col.addStretch(1)
        for text, tip, slot in (
            (">", "Add highlighted datasets", self._add_selected),
            (">>", "Add all datasets", self._add_all),
            ("<", "Remove highlighted datasets", self._remove_selected),
            ("<<", "Remove all datasets", self._remove_all),
        ):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.setFixedWidth(36)
            btn.clicked.connect(slot)
            btn_col.addWidget(btn)
        btn_col.addStretch(1)

        # Right column: datasets to plot, in the order they were added.
        sel_col = QVBoxLayout()
        sel_col.addWidget(QLabel("Plotted (in order)"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.selected_list.itemDoubleClicked.connect(lambda _i: self._remove_selected())
        sel_col.addWidget(self.selected_list)

        # Far right: reorder the Plotted column. Colours stay tied to
        # position, so the dataset on top always gets the first colour.
        order_col = QVBoxLayout()
        order_col.addStretch(1)
        for text, tip, step in (("▲", "Move highlighted dataset up", -1),
                                ("▼", "Move highlighted dataset down", +1)):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda _c=False, s=step: self._move_selected(s))
            order_col.addWidget(btn)
        order_col.addStretch(1)

        group_layout.addLayout(avail_col)
        group_layout.addLayout(btn_col)
        group_layout.addLayout(sel_col)
        group_layout.addLayout(order_col)
        controls.addWidget(group)

        ctrl_group = QGroupBox("Extraction")
        ctrl_layout = QFormLayout(ctrl_group)
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(*self.value_range)
        self.value_spin.setDecimals(self.value_decimals)
        self.value_spin.setValue(self.value_default)
        self.value_label_widget = QLabel(self._value_label_text())
        ctrl_layout.addRow(self.value_label_widget, self.value_spin)

        extract_btn = QPushButton("Extract")
        extract_btn.clicked.connect(self.extract)
        ctrl_layout.addRow(extract_btn)

        export_btn = QPushButton("Export Current Slice...")
        export_btn.clicked.connect(self._export)
        ctrl_layout.addRow(export_btn)

        controls.addWidget(ctrl_group)
        controls.addStretch(1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMaximumWidth(460)

        self.canvas = MplCanvas(figsize=(8, 6))

        root.addWidget(controls_widget)
        root.addWidget(self.canvas, stretch=1)

    # --- dataset columns ----------------------------------------------------

    @staticmethod
    def _labels_in(list_widget):
        return [list_widget.item(i).text() for i in range(list_widget.count())]

    def _set_columns(self, selected):
        """Rebuild both columns. ``selected`` is the ordered plotted list;
        everything else processed goes to the Available column (in the
        processed order). Plotted entries get a colour swatch matching the
        default Matplotlib colour cycle, in order.
        """
        all_labels = self.app_state.processed_labels()
        selected = [l for l in selected if l in all_labels]

        self.available_list.clear()
        for label in all_labels:
            if label not in selected:
                self.available_list.addItem(label)

        self.selected_list.clear()
        for i, label in enumerate(selected):
            item = QListWidgetItem(label)
            item.setIcon(self._swatch(DEFAULT_COLORS[i % len(DEFAULT_COLORS)]))
            self.selected_list.addItem(item)

    @staticmethod
    def _swatch(color):
        pix = QPixmap(12, 12)
        pix.fill(QColor(color))
        return QIcon(pix)

    def _refresh_dataset_list(self):
        # Keep whatever was already added (and still exists); start empty
        # the first time.
        self._set_columns(self._labels_in(self.selected_list))

    def _add_selected(self):
        # Add in the order they appear in the Available column.
        picked = [self.available_list.item(i).text()
                  for i in range(self.available_list.count())
                  if self.available_list.item(i).isSelected()]
        if picked:
            self._set_columns(self._labels_in(self.selected_list) + picked)

    def _add_all(self):
        self._set_columns(self._labels_in(self.selected_list)
                          + self._labels_in(self.available_list))

    def _remove_selected(self):
        keep = [self.selected_list.item(i).text()
                for i in range(self.selected_list.count())
                if not self.selected_list.item(i).isSelected()]
        self._set_columns(keep)

    def _remove_all(self):
        self._set_columns([])

    def _move_selected(self, step):
        """Move the highlighted Plotted entries up (step=-1) or down (+1).
        Colours follow position, so the colour order itself never changes.
        """
        labels = self._labels_in(self.selected_list)
        rows = [i for i in range(len(labels)) if self.selected_list.item(i).isSelected()]
        if not rows:
            return
        if (step < 0 and rows[0] == 0) or (step > 0 and rows[-1] == len(labels) - 1):
            return  # already at the edge
        for r in (rows if step < 0 else reversed(rows)):
            labels[r], labels[r + step] = labels[r + step], labels[r]
        self._set_columns(labels)
        for r in rows:
            self.selected_list.item(r + step).setSelected(True)
        # Redraw the current slice in the new order, if one is shown.
        if self._last_series:
            self.extract()

    def _checked_labels(self):
        """Datasets to plot, in the order they were added."""
        return self._labels_in(self.selected_list)

    def extract(self):
        labels = self._checked_labels()
        if not labels:
            QMessageBox.information(self, "No datasets selected",
                                     "Add at least one dataset to the Plotted column.")
            return

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        series = {}
        x_values = None
        target = self.value_spin.value()

        for idx, label in enumerate(labels):
            result = self.app_state.get_processed(label)
            if result is None:
                continue
            x, y, actual = self._slice(result, target)
            if x is None:
                continue
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=f"{label}",
                    color=DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
            series[label] = (np.asarray(x), np.asarray(y))  # each dataset keeps its own x
            x_values = x

        self._finalize_axes(ax, target)
        self.app_state.apply_ranges_to_axes(ax, x_kind=self.x_range_kind)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

        self._last_x = x_values
        self._last_series = series

    def _apply_ranges(self):
        """Re-apply the global display ranges to the current slice axes."""
        axes = self.canvas.figure.axes
        if not axes:
            return
        self.app_state.apply_ranges_to_axes(axes[0], x_kind=self.x_range_kind)
        self.canvas.draw()

    def _export(self):
        if self._last_x is None or not self._last_series:
            QMessageBox.information(self, "Nothing to export", "Run 'Extract' first.")
            return

        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Slice", "", "CSV (*.csv);;Text (*.txt);;Excel (*.xlsx)")
        if not path:
            return

        fmt = "csv"
        if path.lower().endswith(".txt"):
            fmt = "txt"
        elif path.lower().endswith(".xlsx"):
            fmt = "excel"

        x_min, x_max = self._visible_x_range()
        clipped = {}
        for label, (x, y) in self._last_series.items():
            mask = (x >= x_min) & (x <= x_max)
            if np.any(mask):
                clipped[label] = (x[mask], y[mask])
        if not clipped:
            QMessageBox.information(self, "Nothing to export",
                                     "No data points fall inside the displayed range.")
            return

        try:
            x_label = (f"field ({self.app_state.field_unit})"
                       if self.x_range_kind == "field" else self.x_label)
            export_slices_xy(clipped, path, fmt=fmt, x_label=x_label)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(
            self, "Export complete",
            f"Slice exported to:\n{path}\n\n"
            f"Range exported: {x_min:g} to {x_max:g}")

    def _visible_x_range(self):
        """The x-range currently shown on the plot: the global display range
        (or any toolbar zoom on top of it). Falls back to all data.
        """
        axes = self.canvas.figure.axes
        if axes:
            lo, hi = axes[0].get_xlim()
            return min(lo, hi), max(lo, hi)
        return -np.inf, np.inf

    # --- field unit ---------------------------------------------------------

    value_is_field = False  # True when the spinbox value is a magnetic field

    def _value_label_text(self):
        suffix = f" ({self.app_state.field_unit})" if self.value_is_field else self.value_suffix
        return f"{self.value_label}{suffix}:"

    def _on_field_unit_changed(self, ratio):
        if self.value_is_field:
            rescale_field_spin(self.value_spin, ratio)
        self.value_label_widget.setText(self._value_label_text())
        # Redraw the current slice in the new unit (data are already converted).
        if self._last_series and self._checked_labels():
            self.extract()

    # --- subclass hooks ---------------------------------------------------

    def _slice(self, result, target):
        raise NotImplementedError

    def _finalize_axes(self, ax, target):
        raise NotImplementedError


class FrequencySliceWidget(_BaseSliceWidget):
    """Sub-tab A: extract Signal vs Magnetic Field at a chosen frequency."""

    x_label = "field"
    x_range_kind = "field"
    value_label = "Frequency"
    value_suffix = " (GHz)"
    value_range = (0.0, 1000.0)
    value_default = 8.5
    value_decimals = 3

    def _slice(self, result, target_freq):
        H, sig, actual_freq = result.field_slice_at_frequency(target_freq)
        return H, sig, actual_freq

    def _finalize_axes(self, ax, target):
        ax.set_xlabel(self.app_state.field_axis_label())
        ax.set_ylabel("Signal")
        ax.set_title(f"Frequency Slice near {target:g} GHz")


class FieldSliceWidget(_BaseSliceWidget):
    """Sub-tab B: extract Signal vs Frequency at a chosen magnetic field."""

    x_label = "frequency_GHz"
    x_range_kind = "frequency"
    value_label = "Field"
    value_suffix = ""
    value_is_field = True
    value_range = (-1e6, 1e6)
    value_default = 0.1
    value_decimals = 6

    def _slice(self, result, target_field):
        freqs, vals, actual_field = result.frequency_slice_at_field(target_field)
        return freqs, vals, actual_field

    def _finalize_axes(self, ax, target):
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Signal")
        ax.set_title(f"Field Slice near {target:g} {self.app_state.field_unit}")


class SectionTab(QWidget):
    """Container widget with two sub-tabs: Frequency Slice, Field Slice."""

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        sub_tabs = QTabWidget()

        self.freq_slice_widget = FrequencySliceWidget(app_state)
        self.field_slice_widget = FieldSliceWidget(app_state)

        sub_tabs.addTab(self.freq_slice_widget, "Frequency Slice")
        sub_tabs.addTab(self.field_slice_widget, "Field Slice")

        layout.addWidget(sub_tabs)
