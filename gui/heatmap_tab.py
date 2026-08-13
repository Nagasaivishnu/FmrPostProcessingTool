"""
gui/heatmap_tab.py

Tab 3: Heatmap Visualization.

Shows one processed dataset's full 2D (field x frequency) map at a time,
selected from a dropdown of all processed dataset labels. Controls for
colormap, manual/auto scaling, and interpolation. Zoom/pan/save come for
free from the embedded Matplotlib NavigationToolbar; a crosshair readout
is built into MplCanvas.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from plotting.mpl_canvas import MplCanvas

COLORMAPS = ["viridis", "plasma", "inferno", "magma", "jet", "RdBu", "RdYlBu_r"]
INTERPOLATIONS = ["nearest", "bilinear", "bicubic"]


class HeatmapTab(QWidget):

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._build_ui()
        self.app_state.processed_changed.connect(self._refresh_dataset_list)
        self.app_state.display_range_changed.connect(self.refresh_plot)
        self.app_state.field_unit_changed.connect(self.refresh_plot)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)

        controls = QVBoxLayout()
        controls.addWidget(self._build_dataset_group())
        controls.addWidget(self._build_display_group())
        controls.addStretch(1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMaximumWidth(320)

        self.canvas = MplCanvas(figsize=(8, 6))

        root.addWidget(controls_widget)
        root.addWidget(self.canvas, stretch=1)

    def _build_dataset_group(self) -> QGroupBox:
        group = QGroupBox("Dataset")
        layout = QVBoxLayout(group)

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self.refresh_plot)
        layout.addWidget(QLabel("Dataset:"))
        layout.addWidget(self.dataset_combo)

        return group

    def _build_display_group(self) -> QGroupBox:
        group = QGroupBox("Heatmap Controls")
        layout = QFormLayout(group)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(COLORMAPS)
        self.colormap_combo.currentTextChanged.connect(self.refresh_plot)
        layout.addRow("Colormap:", self.colormap_combo)

        self.auto_scale_checkbox = QCheckBox("Auto Scale")
        self.auto_scale_checkbox.setChecked(True)
        self.auto_scale_checkbox.toggled.connect(self._toggle_manual_scale)
        layout.addRow(self.auto_scale_checkbox)

        self.min_edit = QLineEdit()
        self.max_edit = QLineEdit()
        self.min_edit.setEnabled(False)
        self.max_edit.setEnabled(False)
        layout.addRow("Min:", self.min_edit)
        layout.addRow("Max:", self.max_edit)

        self.interp_combo = QComboBox()
        self.interp_combo.addItems(INTERPOLATIONS)
        self.interp_combo.currentTextChanged.connect(self.refresh_plot)
        layout.addRow("Interpolation:", self.interp_combo)

        refresh_btn = QPushButton("Refresh Plot")
        refresh_btn.clicked.connect(self.refresh_plot)
        layout.addRow(refresh_btn)

        return group

    # --------------------------------------------------------------- logic

    def _toggle_manual_scale(self, auto_checked: bool):
        self.min_edit.setEnabled(not auto_checked)
        self.max_edit.setEnabled(not auto_checked)
        self.refresh_plot()

    def _refresh_dataset_list(self):
        current = self.dataset_combo.currentText()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(self.app_state.processed_labels())
        if current in self.app_state.processed_labels():
            self.dataset_combo.setCurrentText(current)
        self.dataset_combo.blockSignals(False)
        self.refresh_plot()

    def refresh_plot(self, *_args):
        label = self.dataset_combo.currentText()
        result = self.app_state.get_processed(label) if label else None

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        if result is None or not result.sorted_frequencies:
            ax.text(0.5, 0.5, "No processed data.\nRun 'Process All Datasets' in Tab 2.",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            return

        field_grid, freqs, matrix = result.as_matrix()

        vmin = vmax = None
        if not self.auto_scale_checkbox.isChecked():
            try:
                vmin = float(self.min_edit.text()) if self.min_edit.text() else None
                vmax = float(self.max_edit.text()) if self.max_edit.text() else None
            except ValueError:
                vmin = vmax = None  # fall back to auto-scale silently on bad input

        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap=self.colormap_combo.currentText(),
            interpolation=self.interp_combo.currentText(),
            extent=[field_grid.min() * self.app_state.field_scale(),
                    field_grid.max() * self.app_state.field_scale(),
                    freqs.min(), freqs.max()],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(self.app_state.field_axis_label(), fontsize=15)
        ax.set_ylabel("Frequency (GHz)", fontsize=15)
        ax.set_title(f"FMR Map: {label}", fontsize=15)
        cbar = self.canvas.figure.colorbar(im, ax=ax)
        cbar.set_label("Intensity", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Global display ranges: x = field, y = frequency.
        self.app_state.apply_ranges_to_axes(ax, x_kind="field", y_kind="frequency")

        self.canvas.figure.tight_layout()
        self.canvas.draw()
