"""
gui/manual_curves.py

"Manual Curves" panel for the Peak Analysis tab: the user types their own
dispersion-model coefficients and the curves are overlaid on the detected
peak positions (no fitting involved).

Each manual curve is independent:
  - its own model (Kittel, Kittel + H_k, general sqrt, A*sqrt(H+B));
  - coefficient boxes that change to match the chosen model; values typed
    for each model are remembered per curve, so switching back and forth
    doesn't lose them;
  - its own legend name, optional field range, and show/hide tick box.

Coefficients follow the current field unit (T / mT / Oe) and are converted
automatically when the unit is switched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout,
    QWidget,
)

from processing.curve_fitting import FIT_MODELS, FIT_MODELS_BY_KEY, DEFAULT_MODEL_KEY

# Line colours for manual curves (distinct from the dataset colours, drawn dashed).
MANUAL_COLORS = ["black", "#e6194b", "#3cb44b", "#4363d8", "#f58231",
                 "#911eb4", "#808000", "#008080"]

# How each coefficient scales with the field unit: value_new = value_old * ratio**power
# (ratio = new unit / old unit, e.g. 1e4 for T -> Oe).
PARAM_FIELD_POWER = {
    "gamma_mu0_2pi": -1.0,   # GHz per field unit
    "M_eff": 1.0,
    "H_k": 1.0,
    "a": -2.0,               # f^2 = a H^2 + b H + c
    "b": -1.0,
    "c": 0.0,
    "A": -0.5,               # f = A sqrt(H + B)
    "B": 1.0,
}

# Sensible starting values in TESLA (converted to the current unit on use).
DEFAULTS_T = {
    "kittel_inplane": {"gamma_mu0_2pi": 28.0, "M_eff": 1.0},
    "kittel_inplane_aniso": {"gamma_mu0_2pi": 28.0, "M_eff": 1.0, "H_k": 0.0},
    "general_sqrt_quadratic": {"a": 784.0, "b": 784.0, "c": 0.0},
    "sqrt_offset": {"A": 28.0, "B": 0.0},
}


def _param_unit(name: str, unit: str) -> str:
    return {
        "gamma_mu0_2pi": f"GHz/{unit}",
        "M_eff": unit,
        "H_k": unit,
        "a": f"GHz²/{unit}²",
        "b": f"GHz²/{unit}",
        "c": "GHz²",
        "A": f"GHz/{unit}½",
        "B": unit,
    }.get(name, "")


def _default_values(model_key: str, field_factor: float) -> Dict[str, float]:
    base = DEFAULTS_T.get(model_key, {})
    return {k: v * field_factor ** PARAM_FIELD_POWER.get(k, 0.0) for k, v in base.items()}


@dataclass
class ManualCurve:
    name: str
    model_key: str
    # model_key -> {param: value}; remembers what was typed for every model
    values_by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)
    h_min: Optional[float] = None
    h_max: Optional[float] = None
    visible: bool = True

    @property
    def model(self):
        return FIT_MODELS_BY_KEY[self.model_key]

    def params(self) -> List[float]:
        vals = self.values_by_model.get(self.model_key, {})
        return [vals.get(p, np.nan) for p in self.model.param_names]

    def evaluate(self, H: np.ndarray) -> np.ndarray:
        return self.model.func(np.asarray(H, dtype=float), *self.params())

    def rescale(self, ratio: float) -> None:
        for vals in self.values_by_model.values():
            for k in list(vals):
                vals[k] *= ratio ** PARAM_FIELD_POWER.get(k, 0.0)
        if self.h_min is not None:
            self.h_min *= ratio
        if self.h_max is not None:
            self.h_max *= ratio


def _fmt(v: float) -> str:
    if v is None or not np.isfinite(v):
        return ""
    return f"{v:.6g}"


class ManualCurvesPanel(QGroupBox):
    """List of user-defined curves + an editor for the highlighted one."""

    changed = pyqtSignal()  # any curve added/removed/edited

    def __init__(self, app_state, parent=None):
        super().__init__("Manual Curves (your coefficients)", parent)
        self.app_state = app_state
        self.curves: List[ManualCurve] = []
        self._coef_edits: Dict[str, QLineEdit] = {}
        self._loading = False
        self._last_fits_provider = None  # callable -> {label: DatasetFit}
        self._build_ui()
        self.app_state.field_unit_changed.connect(self._on_field_unit_changed)
        self._update_editor_enabled()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.curve_list = QListWidget()
        self.curve_list.setMaximumHeight(90)
        self.curve_list.currentRowChanged.connect(self._load_current)
        self.curve_list.itemChanged.connect(self._on_item_checked)
        layout.addWidget(self.curve_list)

        row = QHBoxLayout()
        add_btn = QPushButton("Add Curve")
        add_btn.clicked.connect(self.add_curve)
        dup_btn = QPushButton("Duplicate")
        dup_btn.clicked.connect(self._duplicate_current)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._remove_current)
        for b in (add_btn, dup_btn, rm_btn):
            row.addWidget(b)
        layout.addLayout(row)

        self.editor = QWidget()
        form = QFormLayout(self.editor)
        form.setContentsMargins(0, 0, 0, 0)

        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._on_name_edited)
        form.addRow("Legend name:", self.name_edit)

        self.model_combo = QComboBox()
        for model in FIT_MODELS:
            self.model_combo.addItem(model.label, model.key)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model:", self.model_combo)

        self.formula_label = QLabel()
        self.formula_label.setWordWrap(True)
        self.formula_label.setStyleSheet("color: #333; font-style: italic;")
        form.addRow(self.formula_label)

        # Coefficient boxes are rebuilt to match the chosen model.
        self.coef_box = QWidget()
        self.coef_form = QFormLayout(self.coef_box)
        self.coef_form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.coef_box)

        range_row = QHBoxLayout()
        self.hmin_edit = QLineEdit()
        self.hmax_edit = QLineEdit()
        for e in (self.hmin_edit, self.hmax_edit):
            e.setPlaceholderText("auto")
            e.editingFinished.connect(self._on_editor_changed)
        range_row.addWidget(self.hmin_edit)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.hmax_edit)
        self.range_label = QLabel()
        form.addRow(self.range_label, range_row)

        copy_btn = QPushButton("Copy Coefficients from Last Fit")
        copy_btn.setToolTip("Fill this curve from the first successful fit "
                            "(same model as the fit) as a starting point")
        copy_btn.clicked.connect(self._copy_from_fit)
        form.addRow(copy_btn)

        layout.addWidget(self.editor)

        note = QLabel("Curves are drawn dashed over the peak data; RMSE / R² "
                      "against each peak track appear under Curve Fit Results.")
        note.setStyleSheet("color: #666;")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._update_unit_labels()

    def set_last_fits_provider(self, provider):
        self._last_fits_provider = provider

    # ------------------------------------------------------------ curves

    def add_curve(self, model_key: Optional[str] = None, values=None, name=None):
        model_key = model_key or DEFAULT_MODEL_KEY
        curve = ManualCurve(name=name or f"Manual {len(self.curves) + 1}",
                            model_key=model_key)
        curve.values_by_model[model_key] = (
            dict(values) if values else _default_values(model_key, self.app_state.field_factor))
        self.curves.append(curve)
        self._rebuild_list(select=len(self.curves) - 1)
        self.changed.emit()
        return curve

    def _duplicate_current(self):
        cur = self.current_curve()
        if cur is None:
            return
        copy = ManualCurve(name=f"{cur.name} (copy)", model_key=cur.model_key,
                           values_by_model={k: dict(v) for k, v in cur.values_by_model.items()},
                           h_min=cur.h_min, h_max=cur.h_max, visible=True)
        self.curves.append(copy)
        self._rebuild_list(select=len(self.curves) - 1)
        self.changed.emit()

    def _remove_current(self):
        row = self.curve_list.currentRow()
        if 0 <= row < len(self.curves):
            del self.curves[row]
            self._rebuild_list(select=min(row, len(self.curves) - 1))
            self.changed.emit()

    def current_curve(self) -> Optional[ManualCurve]:
        row = self.curve_list.currentRow()
        return self.curves[row] if 0 <= row < len(self.curves) else None

    def color_of(self, index: int) -> str:
        return MANUAL_COLORS[index % len(MANUAL_COLORS)]

    def _rebuild_list(self, select: int = -1):
        self.curve_list.blockSignals(True)
        self.curve_list.clear()
        for i, c in enumerate(self.curves):
            item = QListWidgetItem(f"{c.name}  [{c.model.label}]")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if c.visible else Qt.Unchecked)
            item.setForeground(Qt.black)
            pix = QPixmap(12, 12)
            pix.fill(QColor(self.color_of(i)))
            item.setIcon(QIcon(pix))
            self.curve_list.addItem(item)
        self.curve_list.blockSignals(False)
        if select >= 0:
            self.curve_list.setCurrentRow(select)
        self._load_current()

    def _on_item_checked(self, item):
        row = self.curve_list.row(item)
        if 0 <= row < len(self.curves):
            self.curves[row].visible = item.checkState() == Qt.Checked
            self.changed.emit()

    # ------------------------------------------------------------ editor

    def _update_editor_enabled(self):
        self.editor.setEnabled(self.current_curve() is not None)

    def _update_unit_labels(self):
        self.range_label.setText(f"Field range ({self.app_state.field_unit}):")

    def _load_current(self, *_args):
        """Show the highlighted curve's settings in the editor."""
        self._update_editor_enabled()
        cur = self.current_curve()
        if cur is None:
            self._rebuild_coef_fields(None)
            return
        self._loading = True
        self.name_edit.setText(cur.name)
        idx = self.model_combo.findData(cur.model_key)
        self.model_combo.setCurrentIndex(idx)
        self.hmin_edit.setText(_fmt(cur.h_min) if cur.h_min is not None else "")
        self.hmax_edit.setText(_fmt(cur.h_max) if cur.h_max is not None else "")
        self._loading = False
        self._rebuild_coef_fields(cur)

    def _rebuild_coef_fields(self, cur: Optional[ManualCurve]):
        """One coefficient box per parameter of the curve's current model."""
        while self.coef_form.rowCount():
            self.coef_form.removeRow(0)
        self._coef_edits = {}
        if cur is None:
            self.formula_label.setText("Add a curve to enter coefficients.")
            return
        model = cur.model
        self.formula_label.setText(f"Formula:  {model.formula}")
        vals = cur.values_by_model.setdefault(
            cur.model_key, _default_values(cur.model_key, self.app_state.field_factor))
        unit = self.app_state.field_unit
        for p in model.param_names:
            edit = QLineEdit(_fmt(vals.get(p, np.nan)))
            edit.setToolTip(f"{p} in {_param_unit(p, unit)} (e.g. 2.8e-3)")
            edit.editingFinished.connect(self._on_editor_changed)
            u = _param_unit(p, unit)
            self.coef_form.addRow(f"{p} ({u}):" if u else f"{p}:", edit)
            self._coef_edits[p] = edit

    def _on_model_changed(self, *_args):
        if self._loading:
            return
        cur = self.current_curve()
        if cur is None:
            return
        cur.model_key = self.model_combo.currentData()
        self._rebuild_coef_fields(cur)  # boxes now match the new model
        self._refresh_item_text()
        self.changed.emit()

    def _on_name_edited(self):
        cur = self.current_curve()
        if cur is None or self._loading:
            return
        cur.name = self.name_edit.text().strip() or cur.name
        self._refresh_item_text()
        self.changed.emit()

    def _refresh_item_text(self):
        row = self.curve_list.currentRow()
        cur = self.current_curve()
        if cur is not None:
            self.curve_list.item(row).setText(f"{cur.name}  [{cur.model.label}]")

    @staticmethod
    def _parse(text: str) -> Optional[float]:
        text = text.strip().replace("−", "-")
        if not text:
            return None
        return float(text)

    def _on_editor_changed(self, *_args):
        cur = self.current_curve()
        if cur is None or self._loading:
            return
        vals = cur.values_by_model.setdefault(cur.model_key, {})
        bad = []
        for p, edit in self._coef_edits.items():
            try:
                v = self._parse(edit.text())
                if v is None:
                    raise ValueError
                vals[p] = v
                edit.setStyleSheet("")
            except ValueError:
                edit.setStyleSheet("background: #ffd6d6;")
                bad.append(p)
        for attr, edit in (("h_min", self.hmin_edit), ("h_max", self.hmax_edit)):
            try:
                setattr(cur, attr, self._parse(edit.text()))
                edit.setStyleSheet("")
            except ValueError:
                edit.setStyleSheet("background: #ffd6d6;")
        self.changed.emit()

    def _copy_from_fit(self):
        cur = self.current_curve()
        fits = self._last_fits_provider() if self._last_fits_provider else {}
        for dfit in (fits or {}).values():
            for pf in dfit.peak_fits:
                if pf.success:
                    cur.model_key = dfit.model.key
                    cur.values_by_model[dfit.model.key] = {
                        p: float(v) for p, v in zip(dfit.model.param_names, pf.params)}
                    self._load_current()
                    self._refresh_item_text()
                    self.changed.emit()
                    return
        QMessageBox.information(self, "No fit yet",
                                "Run 'Fit Peaks' first, then copy its coefficients.")

    # ------------------------------------------------------------ units

    def _on_field_unit_changed(self, ratio: float):
        for c in self.curves:
            c.rescale(ratio)
        self._update_unit_labels()
        self._load_current()
        self.changed.emit()

    # ------------------------------------------------------- plotting/report

    def curve_xy(self, curve: ManualCurve, data_range=None, n: int = 400):
        """(H, f) arrays for plotting. Range: the curve's own min/max if set,
        else the field display range, else the data range, else 0..0.5 T."""
        lo, hi = None, None
        if self.app_state.field_range is not None:
            lo, hi = self.app_state.field_range
        elif data_range is not None:
            lo, hi = data_range
        if lo is None or hi is None or not hi > lo:
            lo, hi = 0.0, 0.5 * self.app_state.field_factor
        lo = curve.h_min if curve.h_min is not None else lo
        hi = curve.h_max if curve.h_max is not None else hi
        if not hi > lo:
            return np.array([]), np.array([])
        H = np.linspace(lo, hi, n)
        with np.errstate(invalid="ignore"):
            f = curve.evaluate(H)
        return H, f

    def compare_text(self, peak_results) -> str:
        """RMSE / R^2 of every visible manual curve against every peak track."""
        visible = [c for c in self.curves if c.visible]
        if not visible:
            return ""
        lines = ["=== Manual curves vs peak data ===",
                 "(RMSE in GHz; R² of measured frequency vs curve)"]
        for c in visible:
            coef = ", ".join(f"{p} = {v:.4g}" for p, v in
                             zip(c.model.param_names, c.params()))
            lines.append(f"{c.name}  [{c.model.label}]  {coef}")
            if not peak_results:
                lines.append("  (run 'Find Peaks' to compare with data)")
            for label, pr in (peak_results or {}).items():
                for i in range(pr.num_peaks):
                    freqs, fields = pr.track(i)
                    freqs, fields = np.asarray(freqs, float), np.asarray(fields, float)
                    ok = np.isfinite(freqs) & np.isfinite(fields)
                    if c.h_min is not None:
                        ok &= fields >= c.h_min
                    if c.h_max is not None:
                        ok &= fields <= c.h_max
                    if ok.sum() < 2:
                        continue
                    pred = c.evaluate(fields[ok])
                    resid = freqs[ok] - pred
                    rmse = float(np.sqrt(np.mean(resid ** 2)))
                    ss_tot = float(np.sum((freqs[ok] - freqs[ok].mean()) ** 2))
                    r2 = 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
                    lines.append(f"  {label} - Peak {i + 1}: RMSE = {rmse:.4g}, "
                                 f"R² = {r2:.4f}, n = {int(ok.sum())}")
        return "\n".join(lines)
