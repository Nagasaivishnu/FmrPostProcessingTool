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

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QTabWidget,
    QVBoxLayout, QWidget,
)

from plotting.mpl_canvas import MplCanvas
from utils.exporters import export_slice


class _BaseSliceWidget(QWidget):
    """Shared scaffolding for the Frequency-Slice and Field-Slice sub-tabs:
    a dataset checklist (multi-select), a value spinbox, Extract + Export
    buttons, and a plot canvas. Subclasses implement the actual slicing
    and axis labelling.
    """

    x_label = "x"
    value_label = "Value"
    value_suffix = ""
    value_range = (-1e6, 1e6)
    value_default = 0.0
    value_decimals = 4

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._last_x = None
        self._last_series = {}
        self._build_ui()
        self.app_state.processed_changed.connect(self._refresh_dataset_list)

    def _build_ui(self):
        root = QHBoxLayout(self)

        controls = QVBoxLayout()

        group = QGroupBox("Datasets")
        group_layout = QVBoxLayout(group)
        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.NoSelection)
        group_layout.addWidget(self.dataset_list)
        controls.addWidget(group)

        ctrl_group = QGroupBox("Extraction")
        ctrl_layout = QFormLayout(ctrl_group)
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(*self.value_range)
        self.value_spin.setDecimals(self.value_decimals)
        self.value_spin.setValue(self.value_default)
        ctrl_layout.addRow(f"{self.value_label}{self.value_suffix}:", self.value_spin)

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
        controls_widget.setMaximumWidth(300)

        self.canvas = MplCanvas(figsize=(8, 6))

        root.addWidget(controls_widget)
        root.addWidget(self.canvas, stretch=1)

    def _refresh_dataset_list(self):
        checked_before = {
            self.dataset_list.item(i).text()
            for i in range(self.dataset_list.count())
            if self.dataset_list.item(i).checkState() == Qt.Checked
        }
        self.dataset_list.clear()
        for label in self.app_state.processed_labels():
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            should_check = (not checked_before) or (label in checked_before)
            item.setCheckState(Qt.Checked if should_check else Qt.Unchecked)
            self.dataset_list.addItem(item)

    def _checked_labels(self):
        return [
            self.dataset_list.item(i).text()
            for i in range(self.dataset_list.count())
            if self.dataset_list.item(i).checkState() == Qt.Checked
        ]

    def extract(self):
        labels = self._checked_labels()
        if not labels:
            QMessageBox.information(self, "No datasets selected",
                                     "Check at least one dataset in the list.")
            return

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        series = {}
        x_values = None
        target = self.value_spin.value()

        for label in labels:
            result = self.app_state.get_processed(label)
            if result is None:
                continue
            x, y, actual = self._slice(result, target)
            if x is None:
                continue
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=f"{label}")
            series[label] = y
            x_values = x  # shared axis across datasets

        self._finalize_axes(ax, target)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

        self._last_x = x_values
        self._last_series = series

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

        try:
            export_slice(self._last_x, self._last_series, path, fmt=fmt, x_label=self.x_label)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(self, "Export complete", f"Slice exported to:\n{path}")

    # --- subclass hooks ---------------------------------------------------

    def _slice(self, result, target):
        raise NotImplementedError

    def _finalize_axes(self, ax, target):
        raise NotImplementedError


class FrequencySliceWidget(_BaseSliceWidget):
    """Sub-tab A: extract Signal vs Magnetic Field at a chosen frequency."""

    x_label = "field"
    value_label = "Frequency"
    value_suffix = " (GHz)"
    value_range = (0.0, 1000.0)
    value_default = 8.5
    value_decimals = 3

    def _slice(self, result, target_freq):
        H, sig, actual_freq = result.field_slice_at_frequency(target_freq)
        return H, sig, actual_freq

    def _finalize_axes(self, ax, target):
        ax.set_xlabel("Magnetic Field")
        ax.set_ylabel("Signal")
        ax.set_title(f"Frequency Slice near {target:g} GHz")


class FieldSliceWidget(_BaseSliceWidget):
    """Sub-tab B: extract Signal vs Frequency at a chosen magnetic field."""

    x_label = "frequency_GHz"
    value_label = "Field"
    value_suffix = ""
    value_range = (-1e6, 1e6)
    value_default = 0.1
    value_decimals = 6

    def _slice(self, result, target_field):
        freqs, vals, actual_field = result.frequency_slice_at_field(target_field)
        return freqs, vals, actual_field

    def _finalize_axes(self, ax, target):
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Signal")
        ax.set_title(f"Field Slice near {target:g}")


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
