"""
gui/export_tab.py

Tab 5: Data Export.

- Export Heatmap: pick a processed dataset, export its full 2D
  (field, frequency, intensity) matrix as CSV / TXT / NPY / MAT.
- Export Figure: re-render the currently selected dataset's heatmap and
  save it as PNG / SVG / PDF / EPS at a chosen DPI.

(Slice export - frequency-cut and field-cut - lives directly in the
Heatmap Sections tab next to the data it produces, since that's where the
user is already looking at the result they want to save.)
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QLabel, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from utils.exporters import export_heatmap, save_figure

HEATMAP_FORMATS = ["csv", "txt", "npy", "mat"]
FIGURE_FORMATS = ["png", "svg", "pdf", "eps"]
DPI_OPTIONS = [300, 600, 1200]


class ExportTab(QWidget):

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._build_ui()
        self.app_state.processed_changed.connect(self._refresh_dataset_list)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.addWidget(self._build_dataset_group())
        root.addWidget(self._build_heatmap_export_group())
        root.addWidget(self._build_figure_export_group())
        root.addStretch(1)

    def _build_dataset_group(self) -> QGroupBox:
        group = QGroupBox("Dataset")
        layout = QFormLayout(group)
        self.dataset_combo = QComboBox()
        layout.addRow("Dataset:", self.dataset_combo)
        return group

    def _build_heatmap_export_group(self) -> QGroupBox:
        group = QGroupBox("Export Heatmap (Field / Frequency / Intensity Matrix)")
        layout = QFormLayout(group)

        self.heatmap_format_combo = QComboBox()
        self.heatmap_format_combo.addItems(HEATMAP_FORMATS)
        layout.addRow("Format:", self.heatmap_format_combo)

        export_btn = QPushButton("Export Heatmap...")
        export_btn.clicked.connect(self._export_heatmap)
        layout.addRow(export_btn)

        return group

    def _build_figure_export_group(self) -> QGroupBox:
        group = QGroupBox("Export Heatmap Figure")
        layout = QFormLayout(group)

        self.figure_format_combo = QComboBox()
        self.figure_format_combo.addItems(FIGURE_FORMATS)
        layout.addRow("Format:", self.figure_format_combo)

        self.dpi_combo = QComboBox()
        self.dpi_combo.addItems([str(d) for d in DPI_OPTIONS])
        layout.addRow("DPI:", self.dpi_combo)

        export_btn = QPushButton("Export Figure...")
        export_btn.clicked.connect(self._export_figure)
        layout.addRow(export_btn)

        note = QLabel("Tip: figures can also be saved directly from the toolbar\n"
                       "(disk icon) on the Heatmap and Section tab plots.")
        note.setStyleSheet("color: #666;")
        layout.addRow(note)

        return group

    def _refresh_dataset_list(self):
        current = self.dataset_combo.currentText()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(self.app_state.processed_labels())
        if current in self.app_state.processed_labels():
            self.dataset_combo.setCurrentText(current)
        self.dataset_combo.blockSignals(False)

    def _selected_result(self):
        label = self.dataset_combo.currentText()
        if not label:
            return None, None
        return label, self.app_state.get_processed(label)

    def _export_heatmap(self):
        label, result = self._selected_result()
        if result is None:
            QMessageBox.information(self, "No data", "Process a dataset first (Tab 2).")
            return

        fmt = self.heatmap_format_combo.currentText()
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Heatmap", f"{label}_heatmap.{fmt}", f"*.{fmt}")
        if not path:
            return

        try:
            export_heatmap(result, path, fmt=fmt)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(self, "Export complete", f"Heatmap exported to:\n{path}")

    def _export_figure(self):
        label, result = self._selected_result()
        if result is None:
            QMessageBox.information(self, "No data", "Process a dataset first (Tab 2).")
            return

        fmt = self.figure_format_combo.currentText()
        dpi = int(self.dpi_combo.currentText())

        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Figure", f"{label}_heatmap.{fmt}", f"*.{fmt}")
        if not path:
            return

        import matplotlib.pyplot as plt

        field_grid, freqs, matrix = result.as_matrix()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="RdYlBu_r",
                        extent=[field_grid.min(), field_grid.max(), freqs.min(), freqs.max()])
        ax.set_xlabel("Magnetic Field")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(f"FMR Map: {label}")
        fig.colorbar(im, ax=ax, label="Intensity")
        fig.tight_layout()

        try:
            save_figure(fig, path, fmt=fmt, dpi=dpi)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        finally:
            plt.close(fig)

        QMessageBox.information(self, "Export complete", f"Figure exported to:\n{path}")
