"""
gui/peak_tab.py

Tab: Peak Analysis.

Takes the processed heatmap data (field x frequency x intensity) and, for
each frequency's field-sweep cross-section, finds the strongest N peaks
(N = user input). The resulting peak field-positions are plotted with
Magnetic Field on the x-axis and Frequency on the y-axis as dash-dot lines
(one "track" per peak rank, per dataset), and can be exported to
CSV/TXT/Excel.

Peaks 2..N can each be given a maximum allowed field gap from peak 1 (the
lowest-field peak); a peak found farther away than its limit is treated
as noise and its position is overlapped onto peak 1's instead of being
plotted as a stray point. One gap setting appears per peak beyond the
first (e.g. Number of Peaks = 3 -> 2 gap settings), each defaulting to
0.05 T.
"""

from __future__ import annotations

import itertools

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)
import matplotlib.pyplot as plt

from plotting.mpl_canvas import MplCanvas
from processing.peak_analysis import DEFAULT_MAX_GAP, compute_peaks, validate_num_peaks
from utils.exporters import export_peak_data

# Distinct line styles per peak rank so overlapping tracks stay readable;
# cycles if num_peaks exceeds this list's length.
_LINESTYLES = ["-.", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]


class PeakTab(QWidget):

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._last_peak_results = {}  # label -> PeakResult, from the most recent "Find Peaks" run
        self._gap_spinboxes = []  # one QDoubleSpinBox per peak beyond the first
        self._build_ui()
        self.app_state.processed_changed.connect(self._refresh_dataset_list)
        self._rebuild_gap_settings(self.num_peaks_spin.value())

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)

        controls = QVBoxLayout()
        controls.addWidget(self._build_dataset_group())
        controls.addWidget(self._build_settings_group())
        controls.addWidget(self._build_gap_group())
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
        self.num_peaks_spin.valueChanged.connect(self._rebuild_gap_settings)
        layout.addRow("Number of Peaks:", self.num_peaks_spin)

        find_btn = QPushButton("Find Peaks")
        find_btn.clicked.connect(self.find_peaks)
        layout.addRow(find_btn)

        export_btn = QPushButton("Export Peak Data...")
        export_btn.clicked.connect(self._export)
        layout.addRow(export_btn)

        return group

    def _build_gap_group(self) -> QGroupBox:
        """Group holding the per-peak 'max allowed gap from Peak 1'
        settings. Its contents are rebuilt whenever Number of Peaks
        changes (one spinbox per peak beyond the first).
        """
        self.gap_group = QGroupBox("Peak Gap Filtering")
        self.gap_layout = QFormLayout(self.gap_group)

        note = QLabel("If a peak is found farther than this from Peak 1,\n"
                       "it's treated as noise and overlapped onto Peak 1.")
        note.setStyleSheet("color: #666;")
        self.gap_layout.addRow(note)

        return self.gap_group

    # --------------------------------------------------------------- logic

    def _rebuild_gap_settings(self, num_peaks: int):
        """Rebuild the gap-setting spinboxes: one per peak beyond the
        first (num_peaks - 1 total), each defaulting to DEFAULT_MAX_GAP,
        preserving any values the user already entered where possible.
        """
        previous_values = [sb.value() for sb in self._gap_spinboxes]

        # Clear all rows except the note label (row 0).
        while self.gap_layout.rowCount() > 1:
            self.gap_layout.removeRow(1)
        self._gap_spinboxes = []

        n_settings = max(num_peaks - 1, 0)
        for i in range(n_settings):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1e6)
            spin.setDecimals(5)
            spin.setSingleStep(0.01)
            value = previous_values[i] if i < len(previous_values) else DEFAULT_MAX_GAP
            spin.setValue(value)
            self.gap_layout.addRow(f"Peak {i + 2} Max Gap (T):", spin)
            self._gap_spinboxes.append(spin)

    def _current_max_gaps(self):
        return [sb.value() for sb in self._gap_spinboxes]

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

        max_gaps = self._current_max_gaps()

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        results = {}
        all_warnings = []

        for label in labels:
            processed = self.app_state.get_processed(label)
            if processed is None:
                continue

            peak_result = compute_peaks(processed, num_peaks, max_gaps=max_gaps)
            results[label] = peak_result
            all_warnings.extend(f"[{label}] {w}" for w in peak_result.warnings)

            color = next(color_cycle)
            for peak_idx in range(num_peaks):
                freqs, fields = peak_result.track(peak_idx)
                if len(freqs) == 0:
                    continue
                linestyle = _LINESTYLES[peak_idx % len(_LINESTYLES)]
                ax.plot(fields, freqs, linestyle=linestyle, marker="o", markersize=3,
                        linewidth=1.5, color=color,
                        label=f"{label} - Peak {peak_idx + 1}")

        ax.set_xlabel("Peak Field Position")
        ax.set_ylabel("Frequency (GHz)")
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
