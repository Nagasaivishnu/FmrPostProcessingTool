"""
gui/peak_tab.py

Tab: Peak Analysis.

Takes the processed heatmap data (field x frequency x intensity) and, for
each frequency's field-sweep cross-section, finds the strongest N peaks
(N = user input). The resulting peak field-positions are plotted against
frequency as dash-dot lines (one "track" per peak rank, per dataset), and
can be exported to CSV/TXT/Excel.
"""

from __future__ import annotations

import itertools

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)
import matplotlib.pyplot as plt

from plotting.mpl_canvas import MplCanvas
from processing.peak_analysis import compute_peaks, validate_num_peaks
from utils.exporters import export_peak_data

# Distinct line styles per peak rank so overlapping tracks stay readable;
# cycles if num_peaks exceeds this list's length.
_LINESTYLES = ["-.", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]


class PeakTab(QWidget):

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._last_peak_results = {}  # label -> PeakResult, from the most recent "Find Peaks" run
        self._build_ui()
        self.app_state.processed_changed.connect(self._refresh_dataset_list)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)

        controls = QVBoxLayout()
        controls.addWidget(self._build_dataset_group())
        controls.addWidget(self._build_settings_group())
        controls.addStretch(1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMaximumWidth(320)

        self.canvas = MplCanvas(figsize=(8, 6))

        root.addWidget(controls_widget)
        root.addWidget(self.canvas, stretch=1)

    def _build_dataset_group(self) -> QGroupBox:
        group = QGroupBox("Datasets")
        layout = QVBoxLayout(group)
        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.NoSelection)
        layout.addWidget(self.dataset_list)
        return group

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("Peak Detection")
        layout = QFormLayout(group)

        self.num_peaks_spin = QSpinBox()
        self.num_peaks_spin.setRange(1, 50)
        self.num_peaks_spin.setValue(1)
        layout.addRow("Number of Peaks:", self.num_peaks_spin)

        find_btn = QPushButton("Find Peaks")
        find_btn.clicked.connect(self.find_peaks)
        layout.addRow(find_btn)

        export_btn = QPushButton("Export Peak Data...")
        export_btn.clicked.connect(self._export)
        layout.addRow(export_btn)

        return group

    # --------------------------------------------------------------- logic

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

    def find_peaks(self):
        labels = self._checked_labels()
        if not labels:
            QMessageBox.information(self, "No datasets selected",
                                     "Check at least one processed dataset in the list.")
            return

        num_peaks = self.num_peaks_spin.value()
        err = validate_num_peaks(num_peaks)
        if err:
            QMessageBox.warning(self, "Invalid setting", err)
            return

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        results = {}
        all_warnings = []

        for label in labels:
            processed = self.app_state.get_processed(label)
            if processed is None:
                continue

            peak_result = compute_peaks(processed, num_peaks)
            results[label] = peak_result
            all_warnings.extend(f"[{label}] {w}" for w in peak_result.warnings)

            color = next(color_cycle)
            for peak_idx in range(num_peaks):
                freqs, fields = peak_result.track(peak_idx)
                if len(freqs) == 0:
                    continue
                linestyle = _LINESTYLES[peak_idx % len(_LINESTYLES)]
                ax.plot(freqs, fields, linestyle=linestyle, marker="o", markersize=3,
                        linewidth=1.5, color=color,
                        label=f"{label} - Peak {peak_idx + 1}")

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Peak Field Position")
        ax.set_title(f"Peak Position vs Frequency (top {num_peaks} peak(s) per cross-section)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

        self._last_peak_results = results

        if all_warnings:
            QMessageBox.warning(self, "Peak detection completed with warnings",
                                 "\n".join(all_warnings[:30]))

    def _export(self):
        if not self._last_peak_results:
            QMessageBox.information(self, "Nothing to export", "Run 'Find Peaks' first.")
            return

        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Peak Data", "", "CSV (*.csv);;Text (*.txt);;Excel (*.xlsx)")
        if not path:
            return

        fmt = "csv"
        if path.lower().endswith(".txt"):
            fmt = "txt"
        elif path.lower().endswith(".xlsx"):
            fmt = "excel"

        try:
            export_peak_data(self._last_peak_results, path, fmt=fmt)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(self, "Export complete", f"Peak data exported to:\n{path}")
