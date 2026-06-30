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

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(self._build_background_group())
        left.addWidget(self._build_conditioning_group())
        left.addWidget(self._build_enhancement_group())
        left.addWidget(self._build_quantity_group())
        left.addLayout(self._build_action_buttons())
        left.addStretch(1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(420)

        right = QVBoxLayout()
        right.addWidget(QLabel("Preview (first loaded dataset, first frequency)"))
        self.preview_canvas = MplCanvas(figsize=(6, 4.5))
        right.addWidget(self.preview_canvas)

        root.addWidget(left_widget)
        root.addLayout(right, stretch=1)

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

        result = process_dataset(entry.dataset, self.app_state.settings)
        if not result.sorted_frequencies:
            QMessageBox.warning(self, "Preview failed", "No frequencies could be processed.")
            return

        freq = result.sorted_frequencies[0]
        H, sig, _ = result.field_slice_at_frequency(freq)

        self.preview_canvas.figure.clear()
        ax = self.preview_canvas.figure.add_subplot(111)
        ax.plot(H, sig, color="#3366cc")
        ax.set_xlabel("Magnetic Field")
        ax.set_ylabel(self.app_state.settings.output_quantity.replace("_", " ").title())
        ax.set_title(f"{entry.label} @ {freq:g} GHz (preview)")
        ax.grid(alpha=0.3)
        self.preview_canvas.figure.tight_layout()
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
