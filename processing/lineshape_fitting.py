"""
processing/lineshape_fitting.py

Multi-peak derivative-Lorentzian lineshape fitting of raw FMR field sweeps.

The raw lock-in signal of a field-modulated FMR measurement is the field
derivative of the absorption, dP/dH. Each resonance contributes a mix of
the derivative of a Lorentzian (antisymmetric component) and the derivative
of the dispersive lineshape (symmetric component). With d = H - H0 and
half-width w, the model fitted here is

    dP/dH(H) = c0 + c1*H + sum_i [ A_i * (-2 w_i^2 d_i) / (d_i^2 + w_i^2)^2
                                 + B_i * (w_i (w_i^2 - d_i^2)) / (d_i^2 + w_i^2)^2 ]

i.e. a linear baseline plus, per peak, amplitudes (A_i, B_i), resonance
field H0_i and half-width w_i.

Fitting strategy ("advanced", to minimise error robustly):

1. *Physical seeding*: the cumulative integral of dP/dH approximates the
   absorption lineshape, whose maxima sit at the resonance fields. Peak
   positions/widths are seeded from that integral (falling back to extrema
   of |signal| and finally to uniform placement).
2. *Variable projection (VARPRO) initialisation*: for fixed nonlinear
   parameters (positions, widths) the model is LINEAR in
   (c0, c1, A_i, B_i), so those are solved exactly by linear least squares
   rather than guessed.
3. *Multi-start bounded refinement*: several jittered restarts of
   scipy.optimize.least_squares (trust-region-reflective, bounds keeping
   H0 in the field span and w positive), first with a robust ``soft_l1``
   loss and then a plain least-squares polish; the restart with the lowest
   sum of squared errors wins.
4. *Model-order selection*: for a requested peak count N, models with
   k = 1..N+1 peaks are all fitted and the one with the minimum error is
   kept (so N = 3 tries 1, 2, 3 and 4 peaks).

Qt-free, like the rest of ``processing/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
    from scipy.signal import find_peaks, peak_widths, savgol_filter
    from scipy.integrate import cumulative_trapezoid
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def dlorentz_components(H: np.ndarray, H0: float, w: float):
    """Return (antisymmetric, symmetric) derivative-Lorentzian basis shapes
    for one peak at resonance ``H0`` with half-width ``w``.
    """
    d = H - H0
    denom = (d * d + w * w) ** 2
    denom = np.where(denom < 1e-300, 1e-300, denom)
    anti = -2.0 * w * w * d / denom          # d/dH of Lorentzian absorption
    sym = w * (w * w - d * d) / denom        # d/dH of dispersive component
    return anti, sym


def model_signal(H: np.ndarray, params: np.ndarray, n_peaks: int) -> np.ndarray:
    """Evaluate the full model. ``params`` layout:
    [c0, c1, A_1, B_1, H0_1, w_1, A_2, B_2, H0_2, w_2, ...]
    """
    H = np.asarray(H, dtype=float)
    out = params[0] + params[1] * H
    for i in range(n_peaks):
        A, B, H0, w = params[2 + 4 * i: 6 + 4 * i]
        anti, sym = dlorentz_components(H, H0, abs(w))
        out = out + A * anti + B * sym
    return out


def model_jacobian(H: np.ndarray, params: np.ndarray, n_peaks: int) -> np.ndarray:
    """Analytic Jacobian of :func:`model_signal` w.r.t. the parameter
    vector, shape (len(H), 2 + 4*n_peaks). Assumes w > 0 (enforced by the
    fit bounds). Using this instead of finite differences speeds the
    nonlinear refinement up by roughly the parameter count.
    """
    H = np.asarray(H, dtype=float)
    n = H.size
    J = np.empty((n, 2 + 4 * n_peaks))
    J[:, 0] = 1.0
    J[:, 1] = H
    for i in range(n_peaks):
        A, B, H0, w = params[2 + 4 * i: 6 + 4 * i]
        w = abs(w)
        d = H - H0
        D = d * d + w * w
        D = np.where(D < 1e-300, 1e-300, D)
        D2 = D * D
        D3 = D2 * D
        anti = -2.0 * w * w * d / D2
        sym = w * (w * w - d * d) / D2
        # partials of the basis shapes
        danti_dH0 = 2.0 * w * w * (D - 4.0 * d * d) / D3
        danti_dw = -4.0 * w * d * (d * d - w * w) / D3
        dsym_dH0 = 2.0 * w * d * (3.0 * w * w - d * d) / D3
        dsym_dw = (6.0 * w * w * d * d - w ** 4 - d ** 4) / D3
        J[:, 2 + 4 * i] = anti
        J[:, 3 + 4 * i] = sym
        J[:, 4 + 4 * i] = A * danti_dH0 + B * dsym_dH0
        J[:, 5 + 4 * i] = A * danti_dw + B * dsym_dw
    return J


@dataclass
class LineshapeFit:
    """Result of fitting one trace."""

    n_peaks: int                       # peaks in the chosen model
    params: np.ndarray                 # model parameter vector (layout above)
    fitted: np.ndarray                 # fitted curve on the input field axis
    sse: float                         # sum of squared errors vs the data
    r_squared: float
    success: bool = False
    message: str = ""

    def peak_table(self) -> List[Tuple[float, float, float, float]]:
        """[(A, B, H0, w), ...] per fitted peak, sorted by H0."""
        rows = []
        for i in range(self.n_peaks):
            A, B, H0, w = self.params[2 + 4 * i: 6 + 4 * i]
            rows.append((float(A), float(B), float(H0), float(abs(w))))
        return sorted(rows, key=lambda r: r[2])


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def _seed_positions_widths(H: np.ndarray, sig: np.ndarray, k: int,
                           rng: np.random.Generator):
    """Seed (positions, widths) for ``k`` peaks.

    Primary: peaks of the cumulative integral of the signal (approximate
    absorption lineshape). Fallbacks: extrema of |smoothed signal|, then
    uniform placement across the span.
    """
    span = float(np.max(H) - np.min(H))
    default_w = max(span / 30.0, 1e-9)

    n = len(sig)
    win = max(5, (n // 25) | 1)
    try:
        smooth = savgol_filter(sig, min(win, (n - 1) | 1), 2) if n >= 7 else sig
    except Exception:
        smooth = sig

    positions: List[float] = []
    widths: List[float] = []

    # 1) absorption-like integral
    try:
        order = np.argsort(H)
        Hs, ss = H[order], smooth[order]
        absorb = cumulative_trapezoid(ss, Hs, initial=0.0)
        absorb = absorb - np.linspace(absorb[0], absorb[-1], len(absorb))  # de-ramp
        mag = np.abs(absorb - np.median(absorb))
        prom = 0.1 * (np.max(mag) - np.min(mag) + 1e-30)
        idx, _ = find_peaks(mag, prominence=prom)
        if len(idx):
            # strongest first
            idx = idx[np.argsort(mag[idx])[::-1]]
            try:
                w_samples = peak_widths(mag, idx, rel_height=0.5)[0]
            except Exception:
                w_samples = np.full(len(idx), 1.0)
            dH = span / max(len(H) - 1, 1)
            for j, i0 in enumerate(idx[:k]):
                positions.append(float(Hs[i0]))
                widths.append(max(float(w_samples[j]) * dH / 2.0, default_w / 3))
    except Exception:
        pass

    # 2) extrema of |signal|
    if len(positions) < k:
        try:
            mag = np.abs(smooth - np.median(smooth))
            idx, _ = find_peaks(mag, prominence=0.05 * (np.max(mag) + 1e-30))
            idx = idx[np.argsort(mag[idx])[::-1]]
            for i0 in idx:
                p = float(H[i0])
                if all(abs(p - q) > default_w for q in positions):
                    positions.append(p)
                    widths.append(default_w)
                if len(positions) >= k:
                    break
        except Exception:
            pass

    # 3) uniform fill
    while len(positions) < k:
        frac = (len(positions) + 1) / (k + 1)
        positions.append(float(np.min(H) + frac * span + rng.normal(0, 0.02 * span)))
        widths.append(default_w)

    return np.array(positions[:k]), np.array(widths[:k])


def _linear_amplitudes(H: np.ndarray, sig: np.ndarray,
                       positions: np.ndarray, widths: np.ndarray):
    """VARPRO step: with positions/widths fixed, solve the linear-in-model
    parameters (c0, c1, A_i, B_i) exactly by least squares. Returns the full
    parameter vector.
    """
    k = len(positions)
    cols = [np.ones_like(H), H]
    for H0, w in zip(positions, widths):
        anti, sym = dlorentz_components(H, H0, w)
        cols += [anti, sym]
    X = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(X, sig, rcond=None)

    params = np.empty(2 + 4 * k)
    params[0], params[1] = coef[0], coef[1]
    for i in range(k):
        params[2 + 4 * i] = coef[2 + 2 * i]       # A_i
        params[3 + 4 * i] = coef[3 + 2 * i]       # B_i
        params[4 + 4 * i] = positions[i]          # H0_i
        params[5 + 4 * i] = widths[i]             # w_i
    return params


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit_lineshape(H, sig, n_peaks: int, n_restarts: int = 2,
                  max_nfev: int = 1200, rng_seed: int = 0) -> LineshapeFit:
    """Fit ``n_peaks`` derivative-Lorentzian peaks (plus linear baseline) to
    one trace. Multi-start bounded least squares seeded by VARPRO; best SSE
    wins. The first start uses a robust ``soft_l1`` stage before the plain
    polish; jittered restarts use the plain loss only (faster).
    """
    H = np.asarray(H, dtype=float)
    sig = np.asarray(sig, dtype=float)
    good = np.isfinite(H) & np.isfinite(sig)
    H, sig = H[good], sig[good]
    n = H.size
    n_par = 2 + 4 * n_peaks

    empty = np.zeros(0)
    if n < n_par + 2:
        return LineshapeFit(n_peaks, empty, np.zeros(n), np.inf, np.nan,
                            success=False,
                            message=f"{n} points is too few for {n_peaks} peak(s)")
    if not _HAS_SCIPY:
        return LineshapeFit(n_peaks, empty, np.zeros(n), np.inf, np.nan,
                            success=False, message="SciPy unavailable")

    span = float(np.max(H) - np.min(H))
    pad = 0.05 * span
    lo = np.full(n_par, -np.inf)
    hi = np.full(n_par, np.inf)
    for i in range(n_peaks):
        lo[4 + 4 * i], hi[4 + 4 * i] = np.min(H) - pad, np.max(H) + pad  # H0
        lo[5 + 4 * i], hi[5 + 4 * i] = span * 1e-4, span                  # w

    def residual(p):
        return model_signal(H, p, n_peaks) - sig

    def jac(p):
        return model_jacobian(H, p, n_peaks)

    rng = np.random.default_rng(rng_seed)
    best_params = None
    best_sse = np.inf

    for start in range(max(1, n_restarts)):
        try:
            pos, wid = _seed_positions_widths(H, sig, n_peaks, rng)
            if start > 0:  # jitter subsequent restarts
                pos = pos + rng.normal(0, 0.03 * span, size=pos.shape)
                wid = np.clip(wid * rng.uniform(0.5, 2.0, size=wid.shape),
                              span * 1e-4, span)
            pos = np.clip(pos, np.min(H) - pad, np.max(H) + pad)

            p0 = _linear_amplitudes(H, sig, pos, wid)
            p0 = np.clip(p0, lo, hi)

            if start == 0:
                # Robust stage guards the primary seed against outliers...
                scale = np.std(sig - model_signal(H, p0, n_peaks)) or 1.0
                res1 = least_squares(residual, p0, jac=jac, bounds=(lo, hi),
                                     method="trf", loss="soft_l1", f_scale=scale,
                                     x_scale="jac", max_nfev=max_nfev // 2)
                p0 = res1.x
            # ... then a plain least-squares refinement (SSE objective).
            res = least_squares(residual, p0, jac=jac, bounds=(lo, hi),
                                method="trf", x_scale="jac", max_nfev=max_nfev)
            sse = float(np.sum(residual(res.x) ** 2))
            if np.isfinite(sse) and sse < best_sse:
                best_sse, best_params = sse, res.x
        except Exception:
            continue

    if best_params is None:
        return LineshapeFit(n_peaks, empty, np.zeros(n), np.inf, np.nan,
                            success=False, message="all fit attempts failed")

    fitted = model_signal(H, best_params, n_peaks)
    ss_tot = float(np.sum((sig - np.mean(sig)) ** 2))
    r2 = 1.0 - best_sse / ss_tot if ss_tot > 0 else np.nan
    return LineshapeFit(n_peaks, np.asarray(best_params, dtype=float), fitted,
                        best_sse, r2, success=True)


def fit_best_lineshape(H, sig, num_peaks: int, n_restarts: int = 3,
                       rng_seed: int = 0) -> LineshapeFit:
    """Fit models with k = 1 .. ``num_peaks`` + 1 peaks and return the one
    with the minimum error (SSE). E.g. ``num_peaks = 3`` tries 1, 2, 3 and
    4 peaks. If every model order fails, the returned fit has
    ``success=False`` (callers should then substitute zeros).
    """
    num_peaks = max(1, int(num_peaks))
    best: Optional[LineshapeFit] = None
    for k in range(1, num_peaks + 2):
        fit = fit_lineshape(H, sig, k, n_restarts=n_restarts, rng_seed=rng_seed + k)
        if not fit.success:
            continue
        if best is None or fit.sse < best.sse:
            best = fit
    if best is None:
        n = len(np.asarray(H))
        return LineshapeFit(0, np.zeros(0), np.zeros(n), np.inf, np.nan,
                            success=False, message="no model order could be fitted")
    return best
