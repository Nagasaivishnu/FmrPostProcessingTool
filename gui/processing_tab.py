"""
gui/processing_tab.py

Tab 2: Processing Settings.

Groups for:
  A. Background processing method
  B. Signal conditioning (DC offset removal, detrend, Savitzky-Golay)
  C. Signal enhancement (exponential / logarithmic / gamma)
  D. Absorption / derivative calculation method

Plus "Preview Processing" (single dataset, single frequency, quick look)
and "Process All Datasets" (runs the full pipeline over every loaded
experiment and stores results in app_state.processed).
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QRadioButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from processing.dataset_processor import process_dataset
from processing.preprocessing import validate_savgol
from plotting.mpl_canvas import MplCanvas


class ProcessingTab(QWidget):

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._build_ui()
        self.app_state.display_range_changed.connect(self._apply_ranges_to_preview)
        self.app_state.datasets_changed.connect(self._refresh_preview_freq_combo)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(self._build_background_group())
        left.addWidget(self._build_conditioning_group())
        left.addWidget(self._build_enhancement_group())
        left.addWidget(self._build_quantity_group())
        left.addWidget(self._build_display_range_group())
        left.addLayout(self._build_action_buttons())
        left.addStretch(1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(420)

        right = QVBoxLayout()
        right.addWidget(self._build_lineshape_group())
        self.preview_label = QLabel("Preview (first loaded dataset)")
        right.addWidget(self.preview_label)
        self.preview_canvas = MplCanvas(figsize=(6, 4.5))
        right.addWidget(self.preview_canvas)

        root.addWidget(left_widget)
        root.addLayout(right, stretch=1)

    def _build_lineshape_group(self) -> QGroupBox:
        """Controls (shown above the preview graph) for the optional
        raw-data lineshape fit: each frequency's raw trace is fitted with a
        sum of derivative-Lorentzian peaks and the fitted curve replaces the
        raw data for every subsequent processing step. Traces that cannot
        be fitted become zeros. For N requested peaks, models with 1..N+1
        peaks are tried and the minimum-error fit is kept.
        """
        group = QGroupBox("Raw-Data Lineshape Fit (derivative Lorentzian)")
        row = QHBoxLayout(group)

        self.lineshape_checkbox = QCheckBox("Enable fit on raw data")
        self.lineshape_checkbox.toggled.connect(self._sync_settings)
        row.addWidget(self.lineshape_checkbox)

        row.addWidget(QLabel("Number of Peaks:"))
        self.lineshape_peaks_spin = QSpinBox()
        self.lineshape_peaks_spin.setRange(1, 10)
        self.lineshape_peaks_spin.setValue(1)
        self.lineshape_peaks_spin.setToolTip(
            "N peaks requested; models with 1..N+1 peaks are fitted and the "
            "minimum-error one is used.")
        self.lineshape_peaks_spin.valueChanged.connect(self._sync_settings)
        row.addWidget(self.lineshape_peaks_spin)

        row.addWidget(QLabel("Preview frequency:"))
        self.preview_freq_combo = QComboBox()
        self.preview_freq_combo.setMinimumWidth(110)
        self.preview_freq_combo.setToolTip(
            "Frequency shown by 'Preview Fit' and 'Preview Processing'.")
        row.addWidget(self.preview_freq_combo)

        preview_fit_btn = QPushButton("Preview Fit")
        preview_fit_btn.setToolTip(
            "Fit only the selected frequency and overlay raw data vs fitted "
            "curve (fast; does not run the rest of the pipeline).")
        preview_fit_btn.clicked.connect(self._preview_fit)
        row.addWidget(preview_fit_btn)

        row.addStretch(1)
        return group

    def _refresh_preview_freq_combo(self):
        """Populate the preview-frequency dropdown from the first loaded
        dataset, preserving the current selection where possible.
        """
        entry = next((e for e in self.app_state.experiments if e.dataset is not None), None)
        prev = self.preview_freq_combo.currentText()
        self.preview_freq_combo.blockSignals(True)
        self.preview_freq_combo.clear()
        if entry is not None:
            for f in entry.dataset.sorted_frequencies:
                self.preview_freq_combo.addItem(f"{f:g}", f)
            idx = self.preview_freq_combo.findText(prev)
            if idx >= 0:
                self.preview_freq_combo.setCurrentIndex(idx)
        self.preview_freq_combo.blockSignals(False)

    def _selected_preview_freq(self, result_or_dataset):
        """The frequency chosen in the dropdown, or the first available."""
        data = self.preview_freq_combo.currentData()
        freqs = list(result_or_dataset.sorted_frequencies)
        if not freqs:
            return None
        if data is None:
            return freqs[0]
        return min(freqs, key=lambda f: abs(f - float(data)))

    def _build_background_group(self) -> QGroupBox:
        group = QGroupBox("A. Background Processing")
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Method (used only if Background Subtraction is enabled in Tab 1):"))

        self.bg_direct_radio = QRadioButton("Direct subtraction")
        self.bg_normalized_radio = QRadioButton("Normalized subtraction")
        self.bg_division_radio = QRadioButton("Division")
        self.bg_direct_radio.setChecked(True)

        for rb in (self.bg_direct_radio, self.bg_normalized_radio, self.bg_division_radio):
            rb.toggled.connect(self._sync_settings)
            layout.addWidget(rb)

        return group

    def _build_conditioning_group(self) -> QGroupBox:
        group = QGroupBox("B. Signal Conditioning")
        layout = QVBoxLayout(group)

        self.dc_checkbox = QCheckBox("Remove DC Offset  (signal - mean(signal))")
        self.dc_checkbox.setChecked(True)
        self.dc_checkbox.toggled.connect(self._sync_settings)
        layout.addWidget(self.dc_checkbox)

        self.detrend_checkbox = QCheckBox("Detrend  (scipy.signal.detrend)")
        self.detrend_checkbox.toggled.connect(self._sync_settings)
        layout.addWidget(self.detrend_checkbox)

        self.savgol_checkbox = QCheckBox("Apply Savitzky-Golay Filter")
        self.savgol_checkbox.toggled.connect(self._sync_settings)
        layout.addWidget(self.savgol_checkbox)

        savgol_form = QFormLayout()
        self.savgol_window_spin = QSpinBox()
        self.savgol_window_spin.setRange(3, 9999)
        self.savgol_window_spin.setSingleStep(2)
        self.savgol_window_spin.setValue(11)
        self.savgol_window_spin.valueChanged.connect(self._sync_settings)

        self.savgol_poly_spin = QSpinBox()
        self.savgol_poly_spin.setRange(0, 9998)
        self.savgol_poly_spin.setValue(3)
        self.savgol_poly_spin.valueChanged.connect(self._sync_settings)

        savgol_form.addRow("Window Length:", self.savgol_window_spin)
        savgol_form.addRow("Polynomial Order:", self.savgol_poly_spin)
        layout.addLayout(savgol_form)

        self.savgol_warning_label = QLabel("")
        self.savgol_warning_label.setStyleSheet("color: #b00020;")
        layout.addWidget(self.savgol_warning_label)

        return group

    def _build_enhancement_group(self) -> QGroupBox:
        group = QGroupBox("C. Signal Enhancement")
        layout = QVBoxLayout(group)

        self.enhance_checkbox = QCheckBox("Exponentialize / Enhance Data (for better visibility)")
        self.enhance_checkbox.toggled.connect(self._sync_settings)
        layout.addWidget(self.enhance_checkbox)

        self.enhance_combo = QComboBox()
        self.enhance_combo.addItems(["Exponential", "Logarithmic", "Gamma"])
        self.enhance_combo.currentIndexChanged.connect(self._sync_settings)
        layout.addWidget(self.enhance_combo)

        form = QFormLayout()
        self.enhance_scale_spin = QDoubleSpinBox()
        self.enhance_scale_spin.setRange(0.001, 1000.0)
        self.enhance_scale_spin.setValue(1.0)
        self.enhance_scale_spin.setSingleStep(0.1)
        self.enhance_scale_spin.valueChanged.connect(self._sync_settings)
        form.addRow("Scale Factor:", self.enhance_scale_spin)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 10.0)
        self.gamma_spin.setValue(0.5)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.valueChanged.connect(self._sync_settings)
        form.addRow("Gamma Value:", self.gamma_spin)

        layout.addLayout(form)
        return group

    def _build_quantity_group(self) -> QGroupBox:
        group = QGroupBox("D. Output Quantity")
        layout = QVBoxLayout(group)

        self.raw_radio = QRadioButton("Raw Signal")
        self.absorption_radio = QRadioButton("Absorption (cumulative integral)")
        self.first_deriv_radio = QRadioButton("First Derivative")
        self.second_deriv_radio = QRadioButton("Second Derivative")
        self.absorption_radio.setChecked(True)

        for rb in (self.raw_radio, self.absorption_radio, self.first_deriv_radio, self.second_deriv_radio):
            rb.toggled.connect(self._sync_settings)
            layout.addWidget(rb)

        return group

    def _build_display_range_group(self) -> QGroupBox:
        """Global frequency/field display ranges applied to the plots in
        every tab (Heatmap, Sections, Peak Analysis, and this preview).
        Each range is independently enabled; when off, that axis auto-scales.
        """
        group = QGroupBox("E. Display Ranges (applied to all tabs)")
        layout = QFormLayout(group)

        self.field_unit_combo = QComboBox()
        self.field_unit_combo.addItems(["Tesla (T)", "Oersted (Oe)"])
        self.field_unit_combo.setToolTip(
            "Magnetic-field display unit for every plot's field axis. Raw "
            "data is in Tesla; selecting Oe multiplies displayed field "
            "values by 10000. Inputs and data exports stay in Tesla.")
        self.field_unit_combo.currentIndexChanged.connect(self._on_field_unit_changed)
        layout.addRow("Field Unit:", self.field_unit_combo)

        self.freq_range_checkbox = QCheckBox("Limit frequency range (GHz)")
        self.freq_range_checkbox.toggled.connect(self._on_display_range_changed)
        layout.addRow(self.freq_range_checkbox)

        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(0.0, 1000.0)
        self.freq_min_spin.setDecimals(3)
        self.freq_min_spin.setValue(2.0)
        self.freq_min_spin.setEnabled(False)
        self.freq_min_spin.valueChanged.connect(self._on_display_range_changed)
        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(0.0, 1000.0)
        self.freq_max_spin.setDecimals(3)
        self.freq_max_spin.setValue(10.0)
        self.freq_max_spin.setEnabled(False)
        self.freq_max_spin.valueChanged.connect(self._on_display_range_changed)
        layout.addRow("Min Frequency:", self.freq_min_spin)
        layout.addRow("Max Frequency:", self.freq_max_spin)

        self.field_range_checkbox = QCheckBox("Limit field range")
        self.field_range_checkbox.toggled.connect(self._on_display_range_changed)
        layout.addRow(self.field_range_checkbox)

        self.field_min_spin = QDoubleSpinBox()
        self.field_min_spin.setRange(-1e6, 1e6)
        self.field_min_spin.setDecimals(6)
        self.field_min_spin.setSingleStep(0.01)
        self.field_min_spin.setValue(0.0)
        self.field_min_spin.setEnabled(False)
        self.field_min_spin.valueChanged.connect(self._on_display_range_changed)
        self.field_max_spin = QDoubleSpinBox()
        self.field_max_spin.setRange(-1e6, 1e6)
        self.field_max_spin.setDecimals(6)
        self.field_max_spin.setSingleStep(0.01)
        self.field_max_spin.setValue(0.3)
        self.field_max_spin.setEnabled(False)
        self.field_max_spin.valueChanged.connect(self._on_display_range_changed)
        layout.addRow("Min Field:", self.field_min_spin)
        layout.addRow("Max Field:", self.field_max_spin)

        return group

    def _on_field_unit_changed(self, *_args):
        unit = "Oe" if "Oe" in self.field_unit_combo.currentText() else "T"
        self.app_state.set_field_unit(unit)

    def _on_display_range_changed(self, *_args):
        """Enable/disable the spinboxes to match their checkboxes, push the
        ranges into app_state, and notify all tabs (which re-apply them).
        """
        freq_on = self.freq_range_checkbox.isChecked()
        self.freq_min_spin.setEnabled(freq_on)
        self.freq_max_spin.setEnabled(freq_on)
        field_on = self.field_range_checkbox.isChecked()
        self.field_min_spin.setEnabled(field_on)
        self.field_max_spin.setEnabled(field_on)

        freq_range = None
        if freq_on:
            lo, hi = self.freq_min_spin.value(), self.freq_max_spin.value()
            if hi > lo:
                freq_range = (lo, hi)

        field_range = None
        if field_on:
            lo, hi = self.field_min_spin.value(), self.field_max_spin.value()
            if hi > lo:
                field_range = (lo, hi)

        self.app_state.set_display_ranges(freq_range, field_range)

    def _build_action_buttons(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        preview_btn = QPushButton("Preview Processing")
        process_btn = QPushButton("Process All Datasets")
        preview_btn.clicked.connect(self._preview_processing)
        process_btn.clicked.connect(self._process_all)
        layout.addWidget(preview_btn)
        layout.addWidget(process_btn)
        return layout

    # --------------------------------------------------------- settings sync

    def _sync_settings(self, *_args):
        """Push every widget's current value into app_state.settings, and
        validate the Savitzky-Golay parameters live.
        """
        s = self.app_state.settings

        if self.bg_normalized_radio.isChecked():
            s.background_method = "normalized"
        elif self.bg_division_radio.isChecked():
            s.background_method = "division"
        else:
            s.background_method = "direct"

        s.remove_dc = self.dc_checkbox.isChecked()
        s.detrend = self.detrend_checkbox.isChecked()
        s.apply_savgol = self.savgol_checkbox.isChecked()
        s.savgol_window = self.savgol_window_spin.value()
        s.savgol_polyorder = self.savgol_poly_spin.value()

        s.use_lineshape_fit = self.lineshape_checkbox.isChecked()
        s.lineshape_num_peaks = self.lineshape_peaks_spin.value()

        if self.enhance_checkbox.isChecked():
            s.enhance_method = self.enhance_combo.currentText().lower()
        else:
            s.enhance_method = "none"
        s.enhance_scale = self.enhance_scale_spin.value()
        s.gamma_value = self.gamma_spin.value()

        if self.raw_radio.isChecked():
            s.output_quantity = "raw"
        elif self.first_deriv_radio.isChecked():
            s.output_quantity = "first_derivative"
        elif self.second_deriv_radio.isChecked():
            s.output_quantity = "second_derivative"
        else:
            s.output_quantity = "absorption"

        err = validate_savgol(s.savgol_window, s.savgol_polyorder)
        self.savgol_warning_label.setText(err or "")

    # -------------------------------------------------------------- actions

    def _validated_settings_or_warn(self) -> bool:
        self._sync_settings()
        s = self.app_state.settings
        if s.apply_savgol:
            err = validate_savgol(s.savgol_window, s.savgol_polyorder)
            if err:
                QMessageBox.warning(self, "Invalid Savitzky-Golay settings", err)
                return False
        return True

    def _preview_processing(self):
        if not self._validated_settings_or_warn():
            return

        entry = next((e for e in self.app_state.experiments if e.dataset is not None), None)
        if entry is None:
            QMessageBox.information(self, "No data", "Load at least one experimental dataset in Tab 1 first.")
            return

        if not entry.dataset.sorted_frequencies:
            QMessageBox.warning(self, "Preview failed", "The dataset has no frequencies.")
            return

        # Process only the selected frequency (instant even when the
        # raw-data lineshape fit is enabled).
        from processing.dataset_processor import process_record
        freq = self._selected_preview_freq(entry.dataset)
        record = entry.dataset.records[freq]
        try:
            sig, warning = process_record(record, self.app_state.settings)
        except Exception as exc:
            QMessageBox.warning(self, "Preview failed", str(exc))
            return
        H = record.H_field * self.app_state.field_scale()

        self.preview_canvas.figure.clear()
        ax = self.preview_canvas.figure.add_subplot(111)
        ax.plot(H, sig, color="#3366cc")
        ax.set_xlabel(self.app_state.field_axis_label())
        ax.set_ylabel(self.app_state.settings.output_quantity.replace("_", " ").title())
        title = f"{entry.label} @ {freq:g} GHz (preview)"
        if warning:
            title += f"\n{warning}"
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        self.app_state.apply_ranges_to_axes(ax, x_kind="field")
        self.preview_canvas.figure.tight_layout()
        self.preview_canvas.draw()

    def _preview_fit(self):
        """Fit ONLY the selected frequency's raw trace with the multi-peak
        derivative-Lorentzian model and overlay raw data vs fitted curve.
        Fast tuning aid: doesn't run the rest of the pipeline.
        """
        self._sync_settings()
        entry = next((e for e in self.app_state.experiments if e.dataset is not None), None)
        if entry is None:
            QMessageBox.information(self, "No data",
                                     "Load at least one experimental dataset in Tab 1 first.")
            return
        if not entry.dataset.sorted_frequencies:
            QMessageBox.warning(self, "Preview failed", "The dataset has no frequencies.")
            return

        from processing.dataset_processor import select_primary_signal
        from processing.lineshape_fitting import fit_best_lineshape

        freq = self._selected_preview_freq(entry.dataset)
        record = entry.dataset.records[freq]
        raw = select_primary_signal(record)
        fit = fit_best_lineshape(record.H_field, raw,
                                 self.lineshape_peaks_spin.value())

        Hplot = record.H_field * self.app_state.field_scale()
        self.preview_canvas.figure.clear()
        ax = self.preview_canvas.figure.add_subplot(111)
        ax.plot(Hplot, raw, ".", markersize=3, color="0.5", label="raw data")
        if fit.success:
            ax.plot(Hplot, fit.fitted, "-", color="#cc3333", linewidth=1.8,
                    label=f"fit ({fit.n_peaks} peak(s), R\u00b2 = {fit.r_squared:.4f})")
            title = f"{entry.label} @ {freq:g} GHz - lineshape fit"
        else:
            ax.plot(Hplot, fit.fitted, "-", color="#cc3333", linewidth=1.8,
                    label="fit failed -> zeros")
            title = f"{entry.label} @ {freq:g} GHz - fit failed ({fit.message})"
        ax.set_xlabel(self.app_state.field_axis_label())
        ax.set_ylabel("Raw Signal")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        self.app_state.apply_ranges_to_axes(ax, x_kind="field")
        self.preview_canvas.figure.tight_layout()
        self.preview_canvas.draw()

    def _apply_ranges_to_preview(self):
        """Re-apply the global display ranges to the existing preview axes
        (if any) when the ranges change, without reprocessing."""
        axes = self.preview_canvas.figure.axes
        if not axes:
            return
        self.app_state.apply_ranges_to_axes(axes[0], x_kind="field")
        self.preview_canvas.draw()

    def _process_all(self):
        if not self._validated_settings_or_warn():
            return

        loaded = [e for e in self.app_state.experiments if e.dataset is not None]
        if not loaded:
            QMessageBox.information(self, "No data", "Load at least one experimental dataset in Tab 1 first.")
            return

        self.app_state.processed.clear()
        all_warnings = []
        for entry in loaded:
            result = process_dataset(entry.dataset, self.app_state.settings)
            result.label = entry.label
            self.app_state.processed[entry.label] = result
            all_warnings.extend(f"[{entry.label}] {w}" for w in result.warnings)

        self.app_state.processed_changed.emit()

        msg = f"Processed {len(loaded)} dataset(s)."
        if all_warnings:
            msg += "\n\nWarnings:\n" + "\n".join(all_warnings[:30])
            QMessageBox.warning(self, "Processing complete (with warnings)", msg)
        else:
            QMessageBox.information(self, "Processing complete", msg)
