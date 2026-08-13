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
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)
import matplotlib.pyplot as plt

from plotting.mpl_canvas import MplCanvas
from processing.peak_analysis import (
    DEFAULT_MAX_GAP, build_reference_field_curve, compute_peaks, validate_num_peaks,
)
from processing.curve_fitting import (
    DEFAULT_MODEL_KEY, FIT_MODELS, FIT_MODELS_BY_KEY, fit_peak_result, format_fit,
)
from utils.exporters import export_fit_parameters, export_peak_data

# Distinct line styles per peak rank so overlapping tracks stay readable;
# cycles if num_peaks exceeds this list's length.
_LINESTYLES = ["-.", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]


class PeakTab(QWidget):

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._last_peak_results = {}  # label -> PeakResult, from the most recent "Find Peaks" run
        self._last_fits = {}          # label -> DatasetFit, from the most recent "Fit Peaks" run
        self._last_max_field = None   # field cutoff used in the most recent run (or None)
        self._last_ref_cutoff = None  # background ReferenceFieldCutoff used (or None)
        self._gap_spinboxes = []  # one QDoubleSpinBox per peak beyond the first
        self._build_ui()
        self.app_state.processed_changed.connect(self._refresh_dataset_list)
        self._rebuild_gap_settings(self.num_peaks_spin.value())
        self.app_state.display_range_changed.connect(self._reapply_display_ranges)
        self.app_state.field_unit_changed.connect(self._reapply_display_ranges)

    def _reapply_display_ranges(self):
        """Re-render the current peak plot so the new global ranges apply."""
        if self._last_peak_results:
            self._render(self._last_peak_results, self._last_fits)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)

        controls = QVBoxLayout()
        controls.addWidget(self._build_dataset_group())
        controls.addWidget(self._build_settings_group())
        controls.addWidget(self._build_reference_group())
        controls.addWidget(self._build_gap_group())
        controls.addWidget(self._build_fit_group())
        controls.addStretch(1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMaximumWidth(340)

        self.canvas = MplCanvas(figsize=(8, 6))

        right = QVBoxLayout()
        right.addWidget(self.canvas, stretch=1)
        right.addWidget(QLabel("Curve Fit Results"))
        self.fit_results_text = QTextEdit()
        self.fit_results_text.setReadOnly(True)
        self.fit_results_text.setMaximumHeight(170)
        self.fit_results_text.setPlaceholderText(
            "Curve-fit results will appear here after you click 'Fit Peaks'.")
        right.addWidget(self.fit_results_text)
        right_widget = QWidget()
        right_widget.setLayout(right)

        root.addWidget(controls_widget)
        root.addWidget(right_widget, stretch=1)

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

        self.max_field_checkbox = QCheckBox("Ignore peaks above field")
        self.max_field_checkbox.setChecked(False)
        self.max_field_checkbox.toggled.connect(self._on_max_field_toggled)
        layout.addRow(self.max_field_checkbox)

        self.max_field_spin = QDoubleSpinBox()
        self.max_field_spin.setRange(0.0, 1e6)
        self.max_field_spin.setDecimals(5)
        self.max_field_spin.setSingleStep(0.01)
        self.max_field_spin.setValue(0.15)
        self.max_field_spin.setEnabled(False)
        self.max_field_spin.setToolTip(
            "Only peaks at or below this field are detected and used for "
            "fitting; peaks above it are ignored.")
        layout.addRow("Max Field (T):", self.max_field_spin)

        find_btn = QPushButton("Find Peaks")
        find_btn.clicked.connect(self.find_peaks)
        layout.addRow(find_btn)

        export_btn = QPushButton("Export Peak Data...")
        export_btn.clicked.connect(self._export)
        layout.addRow(export_btn)

        return group

    def _build_reference_group(self) -> QGroupBox:
        """Separate, per-frequency field cutoff derived from a *background*
        dataset's peak track: at each frequency, only peaks below the
        background peak's field (minus a margin) are detected. Independent
        of the constant Max Field cutoff in the Peak Detection group.
        """
        group = QGroupBox("Background Reference Cutoff (per-frequency)")
        layout = QFormLayout(group)

        self.ref_cutoff_checkbox = QCheckBox("Use background reference cutoff")
        self.ref_cutoff_checkbox.setChecked(False)
        self.ref_cutoff_checkbox.toggled.connect(self._on_ref_cutoff_toggled)
        layout.addRow(self.ref_cutoff_checkbox)

        self.ref_dataset_combo = QComboBox()
        self.ref_dataset_combo.setEnabled(False)
        self.ref_dataset_combo.setToolTip(
            "Dataset whose peak track defines the per-frequency field ceiling "
            "(e.g. a background/reference measurement).")
        layout.addRow("Background dataset:", self.ref_dataset_combo)

        self.ref_peak_spin = QSpinBox()
        self.ref_peak_spin.setRange(1, 50)
        self.ref_peak_spin.setValue(1)
        self.ref_peak_spin.setEnabled(False)
        self.ref_peak_spin.setToolTip("Which peak of the background dataset to use as the boundary.")
        layout.addRow("Reference peak #:", self.ref_peak_spin)

        self.ref_margin_spin = QDoubleSpinBox()
        self.ref_margin_spin.setRange(0.0, 1e6)
        self.ref_margin_spin.setDecimals(5)
        self.ref_margin_spin.setSingleStep(0.005)
        self.ref_margin_spin.setValue(0.0)
        self.ref_margin_spin.setEnabled(False)
        self.ref_margin_spin.setToolTip(
            "Subtracted from the background field so sample peaks are searched "
            "strictly below the background boundary.")
        layout.addRow("Margin below (T):", self.ref_margin_spin)

        return group

    def _on_ref_cutoff_toggled(self, checked):
        for w in ("ref_dataset_combo", "ref_peak_spin", "ref_margin_spin"):
            if hasattr(self, w):
                getattr(self, w).setEnabled(checked)

    def _build_gap_group(self) -> QGroupBox:
        """Group holding the per-peak 'max allowed gap from Peak 1'
        settings. Its contents are rebuilt whenever Number of Peaks
        changes (one spinbox per peak beyond the first).

        A master "Enable Gap Filtering" checkbox toggles the whole
        feature: when unchecked, the per-peak spinboxes are disabled and
        peak detection runs with ``max_gaps=None`` (far peaks are kept as
        their own points rather than being overlapped onto Peak 1).
        """
        self.gap_group = QGroupBox("Peak Gap Filtering")
        self.gap_layout = QFormLayout(self.gap_group)

        self.gap_enabled_checkbox = QCheckBox("Enable Gap Filtering")
        self.gap_enabled_checkbox.setChecked(True)
        self.gap_enabled_checkbox.toggled.connect(self._update_gap_enabled_state)
        self.gap_layout.addRow(self.gap_enabled_checkbox)

        note = QLabel("If a peak is found farther than this from Peak 1,\n"
                       "it's treated as noise and overlapped onto Peak 1.")
        note.setStyleSheet("color: #666;")
        self.gap_layout.addRow(note)

        return self.gap_group

    def _build_fit_group(self) -> QGroupBox:
        """Group for square-root curve fitting of the peak dispersion
        (frequency vs resonance field). Lets the user pick a model, fit the
        currently found peaks, overlay the fitted curves, and export the
        recovered constants.
        """
        group = QGroupBox("Curve Fitting (Dispersion f vs H)")
        layout = QFormLayout(group)

        self.fit_model_combo = QComboBox()
        for model in FIT_MODELS:
            self.fit_model_combo.addItem(model.label, model.key)
        default_idx = self.fit_model_combo.findData(DEFAULT_MODEL_KEY)
        if default_idx >= 0:
            self.fit_model_combo.setCurrentIndex(default_idx)
        self.fit_model_combo.currentIndexChanged.connect(self._update_formula_label)
        layout.addRow("Model:", self.fit_model_combo)

        self.fit_formula_label = QLabel()
        self.fit_formula_label.setWordWrap(True)
        self.fit_formula_label.setStyleSheet("color: #333; font-style: italic;")
        layout.addRow(self.fit_formula_label)

        self.fit_overlay_checkbox = QCheckBox("Overlay fitted curves on plot")
        self.fit_overlay_checkbox.setChecked(True)
        self.fit_overlay_checkbox.toggled.connect(self._on_overlay_toggled)
        layout.addRow(self.fit_overlay_checkbox)

        self.fit_reject_checkbox = QCheckBox("Reject outliers (fit majority trend)")
        self.fit_reject_checkbox.setChecked(True)
        self.fit_reject_checkbox.toggled.connect(self._on_reject_toggled)
        layout.addRow(self.fit_reject_checkbox)

        self.fit_sigma_spin = QDoubleSpinBox()
        self.fit_sigma_spin.setRange(1.0, 10.0)
        self.fit_sigma_spin.setSingleStep(0.5)
        self.fit_sigma_spin.setDecimals(1)
        self.fit_sigma_spin.setValue(3.0)
        self.fit_sigma_spin.setToolTip(
            "Outlier threshold in robust standard deviations. "
            "Lower = more aggressive rejection; higher = keep more points.")
        layout.addRow("Outlier threshold (sigma):", self.fit_sigma_spin)

        fit_btn = QPushButton("Fit Peaks")
        fit_btn.clicked.connect(self.fit_peaks)
        layout.addRow(fit_btn)

        export_fit_btn = QPushButton("Export Fit Parameters...")
        export_fit_btn.clicked.connect(self._export_fits)
        layout.addRow(export_fit_btn)

        self._update_formula_label()
        return group

    def _update_formula_label(self, *_args):
        key = self.fit_model_combo.currentData()
        model = FIT_MODELS_BY_KEY.get(key)
        if model is None:
            self.fit_formula_label.setText("")
            return
        text = f"Formula:  {model.formula}"
        if model.note:
            text += f"\n{model.note}"
        self.fit_formula_label.setText(text)

    def _on_overlay_toggled(self, _checked):
        # Re-render to add/remove fitted curves without recomputing anything.
        if self._last_peak_results:
            self._render(self._last_peak_results, self._last_fits)

    def _on_reject_toggled(self, checked):
        if hasattr(self, "fit_sigma_spin"):
            self.fit_sigma_spin.setEnabled(checked)

    def _on_max_field_toggled(self, checked):
        if hasattr(self, "max_field_spin"):
            self.max_field_spin.setEnabled(checked)

    # --------------------------------------------------------------- logic

    def _rebuild_gap_settings(self, num_peaks: int):
        """Rebuild the gap-setting spinboxes: one per peak beyond the
        first (num_peaks - 1 total), each defaulting to DEFAULT_MAX_GAP,
        preserving any values the user already entered where possible.
        """
        previous_values = [sb.value() for sb in self._gap_spinboxes]

        # Clear all rows except the checkbox (row 0) and note (row 1).
        while self.gap_layout.rowCount() > 2:
            self.gap_layout.removeRow(2)
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

        self._update_gap_enabled_state()

    def _update_gap_enabled_state(self, *_args):
        """Enable/disable the per-peak gap spinboxes to match the master
        'Enable Gap Filtering' checkbox.
        """
        if not hasattr(self, "gap_enabled_checkbox"):
            return
        enabled = self.gap_enabled_checkbox.isChecked()
        for spin in self._gap_spinboxes:
            spin.setEnabled(enabled)

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

        # Keep the background-reference dataset dropdown in sync, preserving
        # the current selection where possible.
        if hasattr(self, "ref_dataset_combo"):
            prev = self.ref_dataset_combo.currentText()
            self.ref_dataset_combo.blockSignals(True)
            self.ref_dataset_combo.clear()
            self.ref_dataset_combo.addItems(self.app_state.processed_labels())
            idx = self.ref_dataset_combo.findText(prev)
            if idx >= 0:
                self.ref_dataset_combo.setCurrentIndex(idx)
            self.ref_dataset_combo.blockSignals(False)

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

        max_gaps = self._current_max_gaps() if self.gap_enabled_checkbox.isChecked() else None
        max_field = self.max_field_spin.value() if self.max_field_checkbox.isChecked() else None

        # Optional per-frequency cutoff from a background reference dataset.
        ref_cutoff = None
        if self.ref_cutoff_checkbox.isChecked():
            bg_label = self.ref_dataset_combo.currentText()
            bg_proc = self.app_state.get_processed(bg_label) if bg_label else None
            if bg_proc is None:
                QMessageBox.warning(self, "Background reference",
                                     "Select a valid background dataset for the reference cutoff.")
                return
            ref_peak_idx = self.ref_peak_spin.value() - 1
            margin = self.ref_margin_spin.value()
            bg_result = compute_peaks(bg_proc, max(num_peaks, ref_peak_idx + 1),
                                      max_gaps=None)
            ref_cutoff = build_reference_field_curve(
                bg_result, peak_index=ref_peak_idx, margin=margin)
            if ref_cutoff is None:
                QMessageBox.warning(
                    self, "Background reference",
                    f"Could not build a reference boundary from '{bg_label}' "
                    f"(peak {ref_peak_idx + 1}). It may have too few detected points.")
                return

        results = {}
        all_warnings = []
        for label in labels:
            processed = self.app_state.get_processed(label)
            if processed is None:
                continue
            mfbf = (ref_cutoff.fields_for(processed.sorted_frequencies)
                    if ref_cutoff is not None else None)
            peak_result = compute_peaks(processed, num_peaks, max_gaps=max_gaps,
                                        max_field=max_field, max_field_by_freq=mfbf)
            results[label] = peak_result
            all_warnings.extend(f"[{label}] {w}" for w in peak_result.warnings)

        self._last_peak_results = results
        self._last_max_field = max_field
        self._last_ref_cutoff = ref_cutoff
        # Re-finding peaks invalidates any previous fit.
        self._last_fits = {}
        self.fit_results_text.clear()

        self._render(results, fits=None)

        if all_warnings:
            QMessageBox.warning(self, "Peak detection completed with warnings",
                                 "\n".join(all_warnings[:30]))

    def _render(self, results, fits=None):
        """Draw peak tracks (markers + dash-dot line per peak rank, one
        color per dataset), and optionally overlay fitted dispersion curves
        when ``fits`` is provided and overlay is enabled.
        """
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        overlay = fits and self.fit_overlay_checkbox.isChecked()
        color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        fscale = self.app_state.field_scale()  # field display unit (T -> Oe)

        for label, peak_result in results.items():
            color = next(color_cycle)
            for peak_idx in range(peak_result.num_peaks):
                freqs, fields = peak_result.track(peak_idx)
                if len(freqs) == 0:
                    continue
                linestyle = _LINESTYLES[peak_idx % len(_LINESTYLES)]
                ax.plot(fields * fscale, freqs, linestyle=linestyle, marker="o", markersize=3,
                        linewidth=1.2, color=color, alpha=0.85,
                        label=f"{label} - Peak {peak_idx + 1}")

            if overlay and label in fits:
                rejected_labeled = False
                for pf in fits[label].peak_fits:
                    if pf.success and pf.H_fit.size:
                        ax.plot(pf.H_fit * fscale, pf.f_fit, linestyle="-", linewidth=2.2,
                                color=color, alpha=0.95, zorder=5,
                                label=f"{label} - Peak {pf.peak_index + 1} fit")
                    if getattr(pf, "H_out", None) is not None and pf.H_out.size:
                        ax.plot(pf.H_out * fscale, pf.f_out, linestyle="none", marker="x",
                                markersize=5, color="0.6", alpha=0.6, zorder=2,
                                label="rejected (outliers)" if not rejected_labeled else None)
                        rejected_labeled = True

        ax.set_xlabel(f"Peak Field Position ({self.app_state.field_unit_label()})")
        ax.set_ylabel("Frequency (GHz)")
        title = "Peak Position vs Frequency"
        if overlay:
            title += " (with curve fit)"
        ax.set_title(title)

        if getattr(self, "_last_max_field", None) is not None:
            ax.axvline(self._last_max_field * fscale, color="0.5", linestyle=":",
                       linewidth=1.5, zorder=1,
                       label=f"field cutoff = {self._last_max_field * fscale:g} "
                             f"{self.app_state.field_unit_label()}")

        ref = getattr(self, "_last_ref_cutoff", None)
        if ref is not None and getattr(ref, "freqs", None) is not None and ref.freqs.size:
            # Boundary is field(freq); plot field on x, frequency on y.
            ax.plot(ref.fields * fscale, ref.freqs, color="0.4", linestyle="--",
                    linewidth=1.5, zorder=1, label="background reference cutoff")

        # Global display ranges: x = field, y = frequency.
        self.app_state.apply_ranges_to_axes(ax, x_kind="field", y_kind="frequency")

        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def fit_peaks(self):
        if not self._last_peak_results:
            QMessageBox.information(self, "No peaks yet", "Run 'Find Peaks' first.")
            return

        model_key = self.fit_model_combo.currentData()
        model = FIT_MODELS_BY_KEY[model_key]
        reject = self.fit_reject_checkbox.isChecked()
        sigma = self.fit_sigma_spin.value()

        fits = {}
        lines = [f"Model:  {model.formula}"]
        if model.note:
            lines.append(model.note)
        if reject:
            lines.append(f"Outlier rejection: ON (sigma = {sigma:g})")
        else:
            lines.append("Outlier rejection: OFF (fitting all points)")
        lines.append("")

        any_success = False
        for label, peak_result in self._last_peak_results.items():
            dataset_fit = fit_peak_result(peak_result, model_key=model_key,
                                          reject_outliers=reject, sigma=sigma)
            fits[label] = dataset_fit
            lines.append(f"=== {label} ===")
            for pf in dataset_fit.peak_fits:
                lines.append("  " + format_fit(model, pf))
                any_success = any_success or pf.success
            lines.append("")

        self._last_fits = fits
        self.fit_results_text.setPlainText("\n".join(lines))
        self._render(self._last_peak_results, fits=fits)

        if not any_success:
            QMessageBox.warning(
                self, "Curve fitting",
                "No peak track could be fitted. Each track needs at least 3 "
                "detected points, and the data must follow the chosen "
                "square-root model. Try a different model or fewer gap "
                "restrictions.")

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

    def _export_fits(self):
        if not self._last_fits:
            QMessageBox.information(self, "Nothing to export",
                                     "Run 'Fit Peaks' first.")
            return

        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Fit Parameters", "", "CSV (*.csv);;Text (*.txt);;Excel (*.xlsx)")
        if not path:
            return

        fmt = "csv"
        if path.lower().endswith(".txt"):
            fmt = "txt"
        elif path.lower().endswith(".xlsx"):
            fmt = "excel"

        try:
            export_fit_parameters(self._last_fits, path, fmt=fmt)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(self, "Export complete", f"Fit parameters exported to:\n{path}")
