"""
processing/preprocessing.py

Signal-conditioning operations applied to a single field-sweep trace:
background subtraction, DC offset removal, detrending, and Savitzky-Golay
smoothing. Each operation is a small, pure function so they can be unit
tested and chained independently of the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import detrend as scipy_detrend
from scipy.signal import savgol_filter


@dataclass
class PreprocessSettings:
    """Container for all user-configurable preprocessing options.

    This mirrors exactly what the Processing Settings tab exposes, so the
    GUI can build one of these per "Process All" click and pass it straight
    into :func:`preprocess_trace`.
    """

    # --- Background processing ---
    use_background: bool = False
    background_method: str = "direct"  # "direct" | "normalized" | "division"

    # --- Signal conditioning ---
    remove_dc: bool = True
    detrend: bool = False
    apply_savgol: bool = False
    savgol_window: int = 11
    savgol_polyorder: int = 3

    # --- Signal enhancement ---
    enhance_method: str = "none"  # "none" | "exponential" | "logarithmic" | "gamma"
    enhance_scale: float = 1.0
    gamma_value: float = 0.5

    # --- Final quantity to compute ---
    output_quantity: str = "absorption"  # "raw" | "absorption" | "first_derivative" | "second_derivative"


def validate_savgol(window: int, polyorder: int) -> Optional[str]:
    """Return an error string if the Savitzky-Golay parameters are invalid,
    otherwise ``None``. Used by the GUI to block "Process" until fixed.
    """
    if window < 3:
        return "Savitzky-Golay window length must be at least 3."
    if window % 2 == 0:
        return "Savitzky-Golay window length must be odd."
    if polyorder >= window:
        return "Savitzky-Golay polynomial order must be less than window length."
    if polyorder < 0:
        return "Savitzky-Golay polynomial order must be non-negative."
    return None


def _interp_to(reference_x: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``y(x)`` onto ``reference_x``, used whenever a
    background trace's field axis doesn't exactly match the experimental
    trace's field axis.
    """
    if x.shape == reference_x.shape and np.allclose(x, reference_x):
        return y
    # np.interp requires x to be increasing; field sweeps may be descending.
    order = np.argsort(x)
    return np.interp(reference_x, x[order], y[order])


def subtract_background(H_field: np.ndarray, signal: np.ndarray,
                         bg_H_field: np.ndarray, bg_signal: np.ndarray,
                         method: str = "direct") -> np.ndarray:
    """Remove a background reference trace from a signal trace.

    Parameters
    ----------
    method : str
        "direct"     -> signal - background
        "normalized" -> (signal - background) / max(|background|)
        "division"   -> signal / background  (background floor-protected)
    """
    bg_on_grid = _interp_to(H_field, bg_H_field, bg_signal)

    if method == "division":
        denom = np.where(np.abs(bg_on_grid) < 1e-12, 1e-12, bg_on_grid)
        return signal / denom
    elif method == "normalized":
        bg_max = np.max(np.abs(bg_on_grid))
        bg_max = bg_max if bg_max > 0 else 1.0
        return (signal - bg_on_grid) / bg_max
    else:  # "direct"
        return signal - bg_on_grid[:len(signal)]


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """Subtract the mean value from a signal."""
    return signal - np.mean(signal)


def apply_detrend(signal: np.ndarray) -> np.ndarray:
    """Remove a linear trend from a signal (scipy.signal.detrend)."""
    return scipy_detrend(signal)


def apply_savgol(signal: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Smooth a signal with a Savitzky-Golay filter.

    Falls back gracefully (returns the input unchanged) if the trace is
    shorter than the requested window, rather than raising and aborting an
    entire batch process.
    """
    if window > len(signal):
        return signal
    return savgol_filter(signal, window, polyorder)


def enhance_signal(signal: np.ndarray, method: str, scale: float = 1.0,
                    gamma_value: float = 0.5) -> np.ndarray:
    """Apply an optional contrast-enhancing transform to a signal/absorption
    trace purely for visualization purposes.

    "exponential" : exp(scale * normalized_signal)
    "logarithmic" : sign(signal) * log1p(scale * |normalized_signal|)
    "gamma"       : sign(signal) * |normalized_signal| ** gamma_value
    """
    if method == "none":
        return signal

    max_abs = np.max(np.abs(signal))
    normalized = signal / max_abs if max_abs > 0 else signal

    if method == "exponential":
        return np.exp(scale * normalized)
    elif method == "logarithmic":
        return np.sign(normalized) * np.log1p(scale * np.abs(normalized))
    elif method == "gamma":
        return np.sign(normalized) * np.power(np.abs(normalized), gamma_value)
    else:
        return signal


def compute_quantity(H_field: np.ndarray, signal: np.ndarray, quantity: str) -> np.ndarray:
    """Compute the final requested quantity from a conditioned signal.

    "raw"               : signal unchanged
    "absorption"        : cumulative trapezoidal integral of the signal
                           over field (classic FMR derivative-signal ->
                           absorption-lineshape recovery)
    "first_derivative"  : dSignal/dH via numpy.gradient
    "second_derivative" : d2Signal/dH2 via numpy.gradient applied twice
    """
    if quantity == "raw":
        return signal
    elif quantity == "absorption":
        from scipy.integrate import cumulative_trapezoid
        return cumulative_trapezoid(signal, H_field, initial=0)
    elif quantity == "first_derivative":
        return np.gradient(signal, H_field)
    elif quantity == "second_derivative":
        first = np.gradient(signal, H_field)
        return np.gradient(first, H_field)
    else:
        return signal


def preprocess_trace(H_field: np.ndarray, signal: np.ndarray,
                      settings: PreprocessSettings,
                      bg_H_field: Optional[np.ndarray] = None,
                      bg_signal: Optional[np.ndarray] = None) -> np.ndarray:
    """Run the full preprocessing pipeline on a single trace, in a fixed,
    physically sensible order:

        1. Background subtraction
        2. DC offset removal
        3. Detrending
        4. Savitzky-Golay smoothing
        5. Quantity computation (raw / absorption / derivatives)
        6. Signal enhancement (visual contrast only, applied last)

    Returns the processed array. Any individual stage is skipped if its
    checkbox is off in ``settings``.
    """
    out = np.asarray(signal, dtype=float).copy()

    if settings.use_background and bg_H_field is not None and bg_signal is not None:
        out = subtract_background(H_field, out, bg_H_field, bg_signal,
                                   method=settings.background_method)

    if settings.remove_dc:
        out = remove_dc_offset(out)

    if settings.detrend:
        out = apply_detrend(out)

    if settings.apply_savgol:
        err = validate_savgol(settings.savgol_window, settings.savgol_polyorder)
        if err is None:
            out = apply_savgol(out, settings.savgol_window, settings.savgol_polyorder)
        # If invalid, silently skip here - the GUI is responsible for
        # blocking "Process" and surfacing the validation error before we
        # ever reach this point.

    out = compute_quantity(H_field, out, settings.output_quantity)

    out = enhance_signal(out, settings.enhance_method, settings.enhance_scale,
                          settings.gamma_value)

    return out
