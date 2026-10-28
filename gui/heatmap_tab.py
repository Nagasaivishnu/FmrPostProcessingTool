"""
gui/heatmap_tab.py

Tab 3: Heatmap Visualization.

Shows one processed dataset's full 2D (field x frequency) map at a time,
selected from a dropdown of all processed dataset labels.

Controls:
  - Colour: colormap (remembered per dataset), reverse, manual/auto
    scaling, interpolation.
  - Figure Style: publication-ready Origin-like look by default (Arial,
    closed frame, inward ticks on all sides, minor ticks), font size,
    bold, labels, title, normalization, aspect ratio.
  - Save Figure: exact width in cm (journal column presets), PNG / TIFF /
    PDF / SVG / EPS, DPI; save the current map or every dataset at once
    with the same display settings. The on-screen preview is scaled to
    look like the saved figure.

All drawing goes through plotting.heatmap_render so the screen, the PNG
batch save and the Export tab produce identical figures.
"""

from __future__ import annotations

import os
import re

from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from matplotlib.backends.backend_agg import FigureCanvasAgg

from plotting.heatmap_render import (
    ASPECT_PRESETS, SAVE_FORMATS, WIDTH_PRESETS_CM, fit_figure_height,
    render_heatmap, save_heatmap_figure,
)
from plotting.mpl_canvas import MplCanvas

COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo", "jet",
    "rainbow", "hot", "afmhot", "gnuplot2", "cubehelix",
    "coolwarm", "seismic", "bwr", "RdBu", "RdYlBu", "Spectral", "PiYG",
    "gray", "Greys", "Blues", "Reds", "Greens", "Purples", "YlOrRd",
]
INTERPOLATIONS = ["nearest", "bilinear", "bicubic"]
DPI_OPTIONS = ["300", "600", "1200"]


def _safe_filename(label: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", label).strip() or "dataset"


class HeatmapTab(QWidget):

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._build_ui()
        self.app_state.processed_changed.connect(self._refresh_dataset_list)
        self.app_state.display_range_changed.connect(self.refresh_plot)
        self.app_state.field_unit_changed.connect(self._on_field_unit_changed)

    @property
    def style(self):
        return self.app_state.heatmap_style

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)

        controls = QVBoxLayout()
        controls.addWidget(self._build_dataset_group())
        controls.addWidget(self._build_colour_group())
        controls.addWidget(self._build_style_group())
        controls.addWidget(self._build_save_group())

        refresh_btn = QPushButton("Refresh Plot")
        refresh_btn.clicked.connect(self.refresh_plot)
        controls.addWidget(refresh_btn)
        controls.addStretch(1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)

        # Scrollable so all controls stay reachable on small screens.
        scroll = QScrollArea()
        scroll.setWidget(controls_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(390)

        self.canvas = MplCanvas(figsize=(8, 6))
        self.canvas.canvas.mpl_connect("resize_event", self._on_canvas_resized)

        root.addWidget(scroll)
        root.addWidget(self.canvas, stretch=1)

    def _build_dataset_group(self) -> QGroupBox:
        group = QGroupBox("Dataset")
        layout = QVBoxLayout(group)

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        layout.addWidget(self.dataset_combo)

        return group

    def _build_colour_group(self) -> QGroupBox:
        group = QGroupBox("Colour")
        layout = QFormLayout(group)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(COLORMAPS)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        layout.addRow("Colormap:", self.colormap_combo)

        self.reverse_cmap_checkbox = QCheckBox("Reverse colormap")
        self.reverse_cmap_checkbox.toggled.connect(self._on_colormap_changed)
        layout.addRow(self.reverse_cmap_checkbox)

        cmap_note = QLabel("Colormap is remembered separately for each dataset.")
        cmap_note.setStyleSheet("color: #666;")
        cmap_note.setWordWrap(True)
        layout.addRow(cmap_note)

        apply_all_btn = QPushButton("Apply Colormap to All Datasets")
        apply_all_btn.clicked.connect(self._apply_colormap_to_all)
        layout.addRow(apply_all_btn)

        self.auto_scale_checkbox = QCheckBox("Auto Scale")
        self.auto_scale_checkbox.setChecked(True)
        self.auto_scale_checkbox.toggled.connect(self._toggle_manual_scale)
        layout.addRow(self.auto_scale_checkbox)

        self.min_edit = QLineEdit()
        self.max_edit = QLineEdit()
        self.min_edit.setEnabled(False)
        self.max_edit.setEnabled(False)
        self.min_edit.editingFinished.connect(self._on_style_changed)
        self.max_edit.editingFinished.connect(self._on_style_changed)
        layout.addRow("Min:", self.min_edit)
        layout.addRow("Max:", self.max_edit)

        self.interp_combo = QComboBox()
        self.interp_combo.addItems(INTERPOLATIONS)
        self.interp_combo.currentTextChanged.connect(self._on_style_changed)
        layout.addRow("Interpolation:", self.interp_combo)

        return group

    def _build_style_group(self) -> QGroupBox:
        group = QGroupBox("Figure Style (publication, Origin-like)")
        layout = QFormLayout(group)

        pub_btn = QPushButton("Reset to Publication Defaults")
        pub_btn.setToolTip("Origin style, Arial 9 pt, regular weight, no title, "
                           "4:3 plot, 8.5 cm single-column width, 600 dpi")
        pub_btn.clicked.connect(self._apply_publication_defaults)
        layout.addRow(pub_btn)

        self.origin_checkbox = QCheckBox("Origin style")
        self.origin_checkbox.setChecked(self.style.origin_style)
        self.origin_checkbox.setToolTip(
            "Arial, closed black frame, ticks pointing inward on all four "
            "sides, minor ticks, line widths matched to the font size")
        self.origin_checkbox.toggled.connect(self._on_style_changed)
        layout.addRow(self.origin_checkbox)

        self.font_spin = QDoubleSpinBox()
        self.font_spin.setRange(5.0, 40.0)
        self.font_spin.setDecimals(1)
        self.font_spin.setSingleStep(0.5)
        self.font_spin.setValue(self.style.font_size)
        self.font_spin.setSuffix(" pt")
        self.font_spin.setToolTip("Size in the saved figure. Journals usually "
                                  "want 7-10 pt at final printed size.")
        self.font_spin.valueChanged.connect(self._on_style_changed)
        layout.addRow("Font size:", self.font_spin)

        self.bold_checkbox = QCheckBox("Bold text")
        self.bold_checkbox.setChecked(self.style.bold_text)
        self.bold_checkbox.toggled.connect(self._on_style_changed)
        layout.addRow(self.bold_checkbox)

        self.title_checkbox = QCheckBox("Show title (dataset name)")
        self.title_checkbox.setChecked(self.style.show_title)
        self.title_checkbox.toggled.connect(self._on_style_changed)
        layout.addRow(self.title_checkbox)

        self.xlabel_edit = QLineEdit(self.style.x_label)
        self.ylabel_edit = QLineEdit(self.style.y_label)
        self.cbar_label_edit = QLineEdit(self.style.cbar_label)
        for edit in (self.xlabel_edit, self.ylabel_edit, self.cbar_label_edit):
            edit.editingFinished.connect(self._on_style_changed)
        layout.addRow("X label:", self.xlabel_edit)
        layout.addRow("Y label:", self.ylabel_edit)
        layout.addRow("Colour bar label:", self.cbar_label_edit)

        self.normalize_checkbox = QCheckBox("Normalize intensity (0 to 1)")
        self.normalize_checkbox.setChecked(self.style.normalize)
        self.normalize_checkbox.toggled.connect(self._on_style_changed)
        layout.addRow(self.normalize_checkbox)

        self.power_checkbox = QCheckBox("Put \u00d710\u207f multiplier in colour bar label")
        self.power_checkbox.setChecked(self.style.cbar_power_in_label)
        self.power_checkbox.setToolTip("Shows e.g. 'Intensity (a.u.) (\u00d710^-10)' "
                                       "instead of a floating '1e-10' above the bar")
        self.power_checkbox.toggled.connect(self._on_style_changed)
        layout.addRow(self.power_checkbox)

        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(list(ASPECT_PRESETS.keys()))
        self.aspect_combo.setToolTip("Width : height of the plot area")
        self.aspect_combo.currentTextChanged.connect(self._on_aspect_preset_changed)
        layout.addRow("Aspect ratio:", self.aspect_combo)

        custom_row = QHBoxLayout()
        self.aspect_w_spin = QDoubleSpinBox()
        self.aspect_h_spin = QDoubleSpinBox()
        for spin, val in ((self.aspect_w_spin, 4.0), (self.aspect_h_spin, 3.0)):
            spin.setRange(0.1, 100.0)
            spin.setDecimals(2)
            spin.setValue(val)
            spin.setEnabled(False)
            spin.valueChanged.connect(self._on_style_changed)
        custom_row.addWidget(self.aspect_w_spin)
        custom_row.addWidget(QLabel(":"))
        custom_row.addWidget(self.aspect_h_spin)
        layout.addRow("Custom W : H:", custom_row)

        return group

    def _build_save_group(self) -> QGroupBox:
        group = QGroupBox("Save Figure")
        layout = QFormLayout(group)

        self.width_combo = QComboBox()
        self.width_combo.addItems(list(WIDTH_PRESETS_CM.keys()))
        self.width_combo.currentTextChanged.connect(self._on_width_preset_changed)
        layout.addRow("Figure width:", self.width_combo)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(2.0, 60.0)
        self.width_spin.setDecimals(1)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.setValue(self.style.width_cm)
        self.width_spin.setSuffix(" cm")
        self.width_spin.setEnabled(False)
        self.width_spin.valueChanged.connect(self._on_style_changed)
        layout.addRow("Custom width:", self.width_spin)

        self.format_combo = QComboBox()
        self.format_combo.addItems(SAVE_FORMATS)
        self.format_combo.setCurrentText(self.style.save_format)
        self.format_combo.setToolTip("PDF / SVG / EPS keep text and axes as "
                                     "vectors (editable in Illustrator / Inkscape)")
        self.format_combo.currentTextChanged.connect(self._on_style_changed)
        layout.addRow("Format:", self.format_combo)

        self.dpi_combo = QComboBox()
        self.dpi_combo.addItems(DPI_OPTIONS)
        self.dpi_combo.setCurrentText(str(self.style.dpi))
        self.dpi_combo.setToolTip("Resolution of PNG/TIFF, and of the colour map "
                                  "image inside PDF/SVG/EPS")
        self.dpi_combo.currentTextChanged.connect(self._on_style_changed)
        layout.addRow("DPI:", self.dpi_combo)

        self.wysiwyg_checkbox = QCheckBox("Preview at saved size (fonts to scale)")
        self.wysiwyg_checkbox.setChecked(self.style.wysiwyg)
        self.wysiwyg_checkbox.setToolTip(
            "Scales text and lines on screen so the plot looks like the saved "
            "figure, whatever the window size")
        self.wysiwyg_checkbox.toggled.connect(self._on_style_changed)
        layout.addRow(self.wysiwyg_checkbox)

        save_one_btn = QPushButton("Save Current Heatmap...")
        save_one_btn.clicked.connect(self._save_current_png)
        layout.addRow(save_one_btn)

        save_all_btn = QPushButton("Save All Datasets...")
        save_all_btn.setToolTip("One file per dataset, all with the current display settings")
        save_all_btn.clicked.connect(self._save_all_pngs)
        layout.addRow(save_all_btn)

        self.size_note = QLabel()
        self.size_note.setStyleSheet("color: #666;")
        self.size_note.setWordWrap(True)
        layout.addRow(self.size_note)

        return group

    # --------------------------------------------------------------- logic

    def _selected_cmap(self) -> str:
        name = self.colormap_combo.currentText()
        return f"{name}_r" if self.reverse_cmap_checkbox.isChecked() else name

    def _load_cmap_for_current_dataset(self):
        """Set the colormap controls to the current dataset's saved choice."""
        cmap = self.app_state.get_heatmap_cmap(self.dataset_combo.currentText())
        reverse = cmap.endswith("_r")
        base = cmap[:-2] if reverse else cmap
        if self.colormap_combo.findText(base) < 0:
            self.colormap_combo.addItem(base)
        for w in (self.colormap_combo, self.reverse_cmap_checkbox):
            w.blockSignals(True)
        self.colormap_combo.setCurrentText(base)
        self.reverse_cmap_checkbox.setChecked(reverse)
        for w in (self.colormap_combo, self.reverse_cmap_checkbox):
            w.blockSignals(False)

    def _on_dataset_changed(self, *_args):
        self._load_cmap_for_current_dataset()
        self.refresh_plot()

    def _on_colormap_changed(self, *_args):
        self.app_state.set_heatmap_cmap(self.dataset_combo.currentText(),
                                        self._selected_cmap())
        self.refresh_plot()

    def _apply_colormap_to_all(self):
        cmap = self._selected_cmap()
        for label in self.app_state.processed_labels():
            self.app_state.set_heatmap_cmap(label, cmap)
        self.refresh_plot()

    def _toggle_manual_scale(self, auto_checked: bool):
        self.min_edit.setEnabled(not auto_checked)
        self.max_edit.setEnabled(not auto_checked)
        self._on_style_changed()

    def _on_aspect_preset_changed(self, *_args):
        custom = ASPECT_PRESETS.get(self.aspect_combo.currentText()) == "custom"
        self.aspect_w_spin.setEnabled(custom)
        self.aspect_h_spin.setEnabled(custom)
        self._on_style_changed()

    def _on_field_unit_changed(self, _ratio):
        """Put the new unit into the X label, e.g. 'Magnetic Field (T)' ->
        'Magnetic Field (Oe)'. A label without a unit gets one appended."""
        text = self.xlabel_edit.text().rstrip()
        unit = self.app_state.field_unit
        new = re.sub(r"\((T|mT|Oe)\)$", f"({unit})", text)
        if new == text and not re.search(r"\((T|mT|Oe)\)$", text):
            new = f"{text} ({unit})" if text else self.app_state.field_axis_label()
        self.xlabel_edit.setText(new)
        self._on_style_changed()

    def _on_width_preset_changed(self, *_args):
        preset = WIDTH_PRESETS_CM.get(self.width_combo.currentText())
        self.width_spin.setEnabled(preset is None)
        if preset is not None:
            self.width_spin.blockSignals(True)
            self.width_spin.setValue(preset)
            self.width_spin.blockSignals(False)
        self._on_style_changed()

    def _apply_publication_defaults(self):
        """Origin-like journal figure: Arial 9 pt regular, closed frame with
        inward ticks, no title, 4:3 plot, 8.5 cm wide, 600 dpi."""
        widgets = (self.origin_checkbox, self.font_spin, self.bold_checkbox,
                   self.title_checkbox, self.power_checkbox, self.aspect_combo,
                   self.width_combo, self.width_spin, self.dpi_combo,
                   self.wysiwyg_checkbox, self.interp_combo)
        for w in widgets:
            w.blockSignals(True)
        self.origin_checkbox.setChecked(True)
        self.font_spin.setValue(9.0)
        self.bold_checkbox.setChecked(False)
        self.title_checkbox.setChecked(False)
        self.power_checkbox.setChecked(True)
        self.aspect_combo.setCurrentText("4:3")
        self.aspect_w_spin.setEnabled(False)
        self.aspect_h_spin.setEnabled(False)
        self.width_combo.setCurrentText("8.5 cm (single column)")
        self.width_spin.setValue(8.5)
        self.width_spin.setEnabled(False)
        self.dpi_combo.setCurrentText("600")
        self.wysiwyg_checkbox.setChecked(True)
        self.interp_combo.setCurrentText("nearest")
        for w in widgets:
            w.blockSignals(False)
        self._on_style_changed()

    def _on_style_changed(self, *_args):
        """Copy every control into the shared HeatmapStyle, then redraw."""
        s = self.style

        s.vmin = s.vmax = None
        if not self.auto_scale_checkbox.isChecked():
            try:
                s.vmin = float(self.min_edit.text()) if self.min_edit.text() else None
                s.vmax = float(self.max_edit.text()) if self.max_edit.text() else None
            except ValueError:
                s.vmin = s.vmax = None  # bad input: fall back to auto-scale
        s.interpolation = self.interp_combo.currentText()

        s.font_size = float(self.font_spin.value())
        s.origin_style = self.origin_checkbox.isChecked()
        s.bold_text = self.bold_checkbox.isChecked()
        s.show_title = self.title_checkbox.isChecked()
        s.normalize = self.normalize_checkbox.isChecked()
        s.cbar_power_in_label = self.power_checkbox.isChecked()
        s.x_label = self.xlabel_edit.text()
        s.y_label = self.ylabel_edit.text()
        s.cbar_label = self.cbar_label_edit.text()

        preset = ASPECT_PRESETS.get(self.aspect_combo.currentText())
        if preset == "custom":
            s.aspect = (self.aspect_w_spin.value(), self.aspect_h_spin.value())
        else:
            s.aspect = preset

        s.width_cm = self.width_spin.value()
        s.dpi = int(self.dpi_combo.currentText())
        s.save_format = self.format_combo.currentText()
        s.wysiwyg = self.wysiwyg_checkbox.isChecked()

        self.refresh_plot()

    def _refresh_dataset_list(self):
        current = self.dataset_combo.currentText()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(self.app_state.processed_labels())
        if current in self.app_state.processed_labels():
            self.dataset_combo.setCurrentText(current)
        self.dataset_combo.blockSignals(False)
        self._load_cmap_for_current_dataset()
        self.refresh_plot()

    def _render(self, fig, label, scale=1.0):
        result = self.app_state.get_processed(label) if label else None
        apply_ranges = lambda ax: self.app_state.apply_ranges_to_axes(
            ax, x_kind="field", y_kind="frequency")
        render_heatmap(fig, result, label, self.app_state.get_heatmap_cmap(label),
                       self.style, apply_ranges=apply_ranges, scale=scale)

    def _screen_scale(self):
        """How much bigger the on-screen figure is than the saved one, so
        fonts/lines can be scaled to look the same (WYSIWYG preview)."""
        if not self.style.wysiwyg:
            return 1.0
        fw, fh = self.canvas.figure.get_size_inches()
        ew, eh = self.style.png_figsize((fw, fh))
        if ew <= 0 or eh <= 0:
            return 1.0
        return max(0.5, min(fw / ew, fh / eh))

    def refresh_plot(self, *_args):
        self._render(self.canvas.figure, self.dataset_combo.currentText(),
                     scale=self._screen_scale())
        self.canvas.draw()
        self._update_size_note()

    def _on_canvas_resized(self, _event):
        if self.style.wysiwyg:
            self.refresh_plot()

    def _update_size_note(self):
        if not hasattr(self, "size_note"):
            return
        s = self.style
        fmt = s.save_format.upper()
        px = "" if s.save_format in ("pdf", "svg", "eps") else \
            f" ({int(s.width_in * s.dpi)} px at {s.dpi} dpi)"
        self.size_note.setText(
            f"Saved as {fmt}, exactly {s.width_cm:g} cm wide{px}; height "
            "follows the aspect ratio. "
            "Uses the settings above, the shared display range and each "
            "dataset's own colormap.")

    # ------------------------------------------------------------ saving

    def _save_png(self, label, path):
        s = self.style
        fig = Figure(figsize=s.png_figsize(self.canvas.figure.get_size_inches()))
        FigureCanvasAgg(fig)
        self._render(fig, label)
        fit_figure_height(fig, s)
        save_heatmap_figure(fig, path, s.save_format, s.dpi)

    def _ext(self):
        return "tif" if self.style.save_format == "tiff" else self.style.save_format

    def _save_current_png(self):
        label = self.dataset_combo.currentText()
        if not label or self.app_state.get_processed(label) is None:
            QMessageBox.information(self, "No data", "Process a dataset first (Tab 2).")
            return
        ext = self._ext()
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save Heatmap", f"{_safe_filename(label)}_heatmap.{ext}",
            f"{ext.upper()} (*.{ext})")
        if not path:
            return
        if not path.lower().endswith(f".{ext}"):
            path += f".{ext}"
        try:
            self._save_png(label, path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Heatmap saved to:\n{path}")

    def _save_all_pngs(self):
        labels = self.app_state.processed_labels()
        if not labels:
            QMessageBox.information(self, "No data", "Process datasets first (Tab 2).")
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose folder for heatmap figures")
        if not folder:
            return

        ext = self._ext()
        paths = {label: os.path.join(folder, f"{_safe_filename(label)}_heatmap.{ext}")
                 for label in labels}
        existing = [p for p in paths.values() if os.path.exists(p)]
        if existing:
            reply = QMessageBox.question(
                self, "Overwrite files?",
                f"{len(existing)} of these files already exist in that folder.\n"
                "Overwrite them?")
            if reply != QMessageBox.Yes:
                return

        saved, failed = [], []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for label, path in paths.items():
                try:
                    self._save_png(label, path)
                    saved.append(os.path.basename(path))
                except Exception as exc:
                    failed.append(f"{label}: {exc}")
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()

        msg = f"Saved {len(saved)} {ext.upper()} file(s) to:\n{folder}"
        if failed:
            msg += "\n\nFailed:\n" + "\n".join(failed)
            QMessageBox.warning(self, "Save finished with errors", msg)
        else:
            QMessageBox.information(self, "Saved", msg)
