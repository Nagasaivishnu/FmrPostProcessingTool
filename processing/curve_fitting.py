"""
processing/curve_fitting.py

Curve fitting for FMR peak-position tracks (frequency vs resonance field).

Theory
------
For an in-plane magnetized film the broadband FMR dispersion follows the
Kittel relation

    f = (gamma * mu0 / 2pi) * sqrt( H * (H + M_eff) )

where ``gamma*mu0/2pi`` is the gyromagnetic prefactor (~28 GHz/T for g=2)
and ``M_eff`` is the effective magnetization (in field units). This module
fits that relation - and a couple of more flexible generalizations of it -
to each detected peak track, reporting the recovered constants, their
uncertainties, a goodness-of-fit (R^2), and (where relevant) physical
constants derived from the raw fit parameters.

Models
------
* ``kittel_inplane`` (default): the exact relation above.
* ``kittel_inplane_aniso``: adds an in-plane anisotropy field H_k,
      f = (gamma*mu0/2pi) * sqrt( (H + H_k) * (H + H_k + M_eff) );
  reduces to the exact relation when H_k = 0.
* ``general_sqrt_quadratic``: the most flexible square-root form,
      f = sqrt( a*H^2 + b*H + c );
  this is linear in f^2 so it always fits by ordinary least squares (no
  convergence issues), and the physical constants are derived from it as
  gamma*mu0/2pi = sqrt(a) and M_eff = b/a.
* ``sqrt_offset``: a simple generic root, f = A*sqrt(H + B).

Design notes
------------
Qt-free, like the rest of ``processing/``. Each model gets a deterministic
initial estimate by linearizing in f^2, then a nonlinear refinement via
:func:`scipy.optimize.curve_fit` to obtain parameter standard errors. If
refinement is unavailable or fails, the linearized estimate is kept so the
feature degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

try:  # SciPy is a project dependency; degrade gracefully if absent.
    from scipy.optimize import curve_fit, least_squares
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# --------------------------------------------------------------------------
# Model functions:  f (frequency, GHz)  as a function of  H (resonance field)
# --------------------------------------------------------------------------

def _f_kittel_inplane(H, gamma, M_eff):
    """f = gamma * sqrt(H * (H + M_eff))  -- exact in-plane Kittel."""
    return gamma * np.sqrt(np.maximum(H * (H + M_eff), 0.0))


def _f_kittel_inplane_aniso(H, gamma, M_eff, H_k):
    """f = gamma * sqrt((H + H_k) * (H + H_k + M_eff))  -- Kittel + in-plane
    anisotropy field H_k.
    """
    Hs = H + H_k
    return gamma * np.sqrt(np.maximum(Hs * (Hs + M_eff), 0.0))


def _f_general_sqrt_quadratic(H, a, b, c):
    """f = sqrt(a*H^2 + b*H + c)  -- most flexible square-root form."""
    return np.sqrt(np.maximum(a * H * H + b * H + c, 0.0))


def _f_sqrt_offset(H, A, B):
    """f = A * sqrt(H + B)  -- simple generic root, B inside the root."""
    return A * np.sqrt(np.maximum(H + B, 0.0))


# --------------------------------------------------------------------------
# Derived-quantity helpers (physical constants computed from raw params)
# --------------------------------------------------------------------------

def _derived_general_quadratic(params) -> Dict[str, float]:
    a, b, c = params
    gamma = np.sqrt(a) if a > 0 else np.nan
    M_eff = b / a if a != 0 else np.nan
    return {"gamma_mu0_2pi": float(gamma), "M_eff": float(M_eff)}


@dataclass(frozen=True)
class FitModel:
    """A fittable dispersion model."""

    key: str
    label: str                       # short name for a dropdown
    formula: str                     # human-readable formula
    param_names: List[str]           # raw fit parameters, in func order
    func: Callable[..., np.ndarray]  # func(H, *params) -> f
    note: str = ""                   # short physical hint
    derived: Optional[Callable[[np.ndarray], Dict[str, float]]] = None


FIT_MODELS: List[FitModel] = [
    FitModel(
        key="kittel_inplane",
        label="Kittel in-plane (theory)",
        formula="f = (gamma*mu0/2pi) * sqrt( H * (H + M_eff) )",
        param_names=["gamma_mu0_2pi", "M_eff"],
        func=_f_kittel_inplane,
        note="gamma*mu0/2pi ~ 28 GHz/T for g=2; M_eff is the effective magnetization (field units).",
    ),
    FitModel(
        key="kittel_inplane_aniso",
        label="Kittel + anisotropy (H_k)",
        formula="f = (gamma*mu0/2pi) * sqrt( (H + H_k) * (H + H_k + M_eff) )",
        param_names=["gamma_mu0_2pi", "M_eff", "H_k"],
        func=_f_kittel_inplane_aniso,
        note="Adds an in-plane anisotropy field H_k; reduces to plain Kittel when H_k = 0.",
    ),
    FitModel(
        key="general_sqrt_quadratic",
        label="General sqrt(a*H^2+b*H+c)",
        formula="f = sqrt( a*H^2 + b*H + c )",
        param_names=["a", "b", "c"],
        func=_f_general_sqrt_quadratic,
        note="Most flexible (fits by linear least squares). Derived: gamma*mu0/2pi = sqrt(a), M_eff = b/a.",
        derived=_derived_general_quadratic,
    ),
    FitModel(
        key="sqrt_offset",
        label="Simple A*sqrt(H+B)",
        formula="f = A * sqrt( H + B )",
        param_names=["A", "B"],
        func=_f_sqrt_offset,
        note="Generic square root; B is the constant inside the root.",
    ),
]

FIT_MODELS_BY_KEY: Dict[str, FitModel] = {m.key: m for m in FIT_MODELS}
DEFAULT_MODEL_KEY = "kittel_inplane"


@dataclass
class PeakFit:
    """Fit result for a single peak track."""

    peak_index: int                 # 0-based; display as peak_index + 1
    model_key: str
    params: np.ndarray              # fitted constants, in model.param_names order
    param_errors: np.ndarray        # 1-sigma standard errors (NaN if unavailable)
    r_squared: float
    n_points: int
    H_fit: np.ndarray = field(default_factory=lambda: np.array([]))
    f_fit: np.ndarray = field(default_factory=lambda: np.array([]))
    derived: Dict[str, float] = field(default_factory=dict)
    n_total: int = 0                # points available before outlier rejection
    n_rejected: int = 0             # points dropped as outliers
    H_out: np.ndarray = field(default_factory=lambda: np.array([]))  # rejected x
    f_out: np.ndarray = field(default_factory=lambda: np.array([]))  # rejected y
    success: bool = False
    message: str = ""


@dataclass
class DatasetFit:
    """All peak fits for one dataset, plus the model they were fit with."""

    label: str
    model: FitModel
    peak_fits: List[PeakFit] = field(default_factory=list)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def _linearized_estimate(H: np.ndarray, f: np.ndarray, model_key: str) -> np.ndarray:
    """Deterministic initial estimate of the parameters by linearizing in
    f^2. Raises ``RuntimeError`` for non-physical estimates.
    """
    f2 = f ** 2

    if model_key == "kittel_inplane":       # f^2 = a*H^2 + a*M_eff*H (no const)
        # Robust form: f^2 / H = gamma^2 * (H + M_eff) is LINEAR in H. A
        # Theil-Sen slope resists any leverage outliers still present after
        # masking and avoids the ill-conditioned through-origin quadratic.
        pos = H > 0
        if np.count_nonzero(pos) >= 2:
            ratio = f2[pos] / H[pos]
            try:
                from scipy.stats import theilslopes
                slope, intercept, *_ = theilslopes(ratio, H[pos])
            except Exception:
                slope, intercept = np.polyfit(H[pos], ratio, 1)
            if slope > 0:
                return np.array([np.sqrt(slope), intercept / slope])
        # Fallback: full quadratic in f^2 (intercept absorbs conditioning).
        X = np.vstack([H ** 2, H, np.ones_like(H)]).T
        a, b, _c = np.linalg.lstsq(X, f2, rcond=None)[0]
        if a <= 0:
            raise RuntimeError("non-positive quadratic term")
        return np.array([np.sqrt(a), b / a])

    if model_key == "general_sqrt_quadratic":   # f^2 = a*H^2 + b*H + c
        X = np.vstack([H ** 2, H, np.ones_like(H)]).T
        coef, *_ = np.linalg.lstsq(X, f2, rcond=None)
        return np.asarray(coef, dtype=float)

    if model_key == "kittel_inplane_aniso":
        # Estimate via the general quadratic, then convert a,b,c -> gamma,M_eff,H_k.
        X = np.vstack([H ** 2, H, np.ones_like(H)]).T
        coef, *_ = np.linalg.lstsq(X, f2, rcond=None)
        a, b, c = coef
        if a <= 0:
            raise RuntimeError("non-positive quadratic term")
        gamma = np.sqrt(a)
        P = b / a                # = 2*H_k + M_eff
        Q = c / a                # = H_k*(H_k + M_eff) -> H_k^2 - P*H_k + Q = 0
        disc = P * P - 4 * Q
        if disc >= 0:
            r = np.sqrt(disc)
            roots = [(P - r) / 2, (P + r) / 2]
            H_k = min(roots, key=abs)        # anisotropy usually small
        else:
            H_k = 0.0
        M_eff = P - 2 * H_k
        return np.array([gamma, M_eff, H_k])

    if model_key == "sqrt_offset":          # f^2 = A^2*H + A^2*B
        m, c = np.polyfit(H, f2, 1)
        if m <= 0:
            raise RuntimeError("non-positive slope")
        return np.array([np.sqrt(m), c / m])

    raise ValueError(f"Unknown model key: {model_key}")


def _robust_scale(r: np.ndarray) -> float:
    """Median-absolute-deviation scale estimate (~sigma), with fallbacks."""
    med = np.median(r)
    s = 1.4826 * np.median(np.abs(r - med))
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(r))
    return s if (np.isfinite(s) and s > 0) else 1.0


def _robust_inlier_mask(H: np.ndarray, f: np.ndarray, sigma: float = 2.5,
                        min_frac: float = 0.3, n_iter: int = 600,
                        rng_seed: int = 0) -> np.ndarray:
    """Boolean mask of points that follow the majority dispersion trend.

    Uses RANSAC consensus in ``f**2`` space (where every square-root model
    is at most quadratic in H), which is robust both to a large *fraction*
    of outliers and to high-*leverage* outliers at extreme field values -
    the situation that ordinary or even Cauchy-weighted least squares
    handles poorly. Many random 3-point quadratics are tried; the one whose
    consensus set (points within ``sigma`` robust deviations) is largest
    wins, then the inlier set is polished with a couple of reclip steps. A
    floor (``min_frac``) prevents over-trimming.
    """
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float)
    n = H.size
    if n < 6:
        return np.ones(n, dtype=bool)

    y = f ** 2
    X = np.vstack([H ** 2, H, np.ones_like(H)]).T

    # Robust global residual scale -> a fixed RANSAC inlier threshold. A
    # Theil-Sen line (robust to outliers and leverage) gives a residual
    # scale dominated by the majority spread, not the outliers.
    try:
        from scipy.stats import theilslopes
        m_ts, c_ts, *_ = theilslopes(y, H)
        scale = _robust_scale(y - (m_ts * H + c_ts))
    except Exception:
        scale = _robust_scale(y - np.median(y))
    thresh = sigma * scale

    rng = np.random.default_rng(rng_seed)
    idx = np.arange(n)
    k = 3
    floor = max(k, int(min_frac * n))

    best_mask = None
    best_count = -1
    for _ in range(n_iter):
        sample = rng.choice(idx, size=k, replace=False)
        try:
            beta, *_ = np.linalg.lstsq(X[sample], y[sample], rcond=None)
        except Exception:
            continue
        mask = np.abs(y - X @ beta) < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask

    if best_mask is None or best_mask.sum() < floor:
        return np.ones(n, dtype=bool)

    # Polish: refit on the consensus set and reclip with the model-residual
    # robust scale, a few concentration steps.
    mask = best_mask
    for _ in range(5):
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        r = y - X @ beta
        med = np.median(r[mask])
        s = _robust_scale(r[mask])
        new_mask = np.abs(r - med) < sigma * s
        if new_mask.sum() < floor or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    return mask


def _robust_model_fit(func, H, f, p0):
    """Nonlinear fit with a Cauchy robust loss (down-weights residual
    outliers/leverage points). Returns (params, param_errors). Falls back
    to the initial guess if SciPy is unavailable or the fit fails.
    """
    p0 = np.asarray(p0, dtype=float)
    if not _HAS_SCIPY:
        return p0, np.full(p0.size, np.nan)
    try:
        scale = _robust_scale(f - func(H, *p0))
        res = least_squares(lambda p: func(H, *p) - f, p0, loss="cauchy",
                            f_scale=max(scale, 1e-9), max_nfev=20000)
        params = res.x
        # Approximate covariance from the Jacobian at the solution.
        dof = max(1, f.size - params.size)
        resid = func(H, *params) - f
        mse = float(np.sum(resid ** 2)) / dof
        try:
            cov = np.linalg.inv(res.jac.T @ res.jac) * mse
            perr = np.sqrt(np.abs(np.diag(cov)))
        except Exception:
            perr = np.full(params.size, np.nan)
        return np.asarray(params, dtype=float), np.asarray(perr, dtype=float)
    except Exception:
        return p0, np.full(p0.size, np.nan)


def fit_track(H, f, model: FitModel, peak_index: int = 0,
              n_curve: int = 200, min_points: Optional[int] = None,
              reject_outliers: bool = True, sigma: float = 3.0) -> PeakFit:
    """Fit one peak track ``f(H)`` to ``model``.

    Parameters
    ----------
    reject_outliers : bool
        If True (default), points far from the majority dispersion trend
        are identified (see :func:`_robust_inlier_mask`) and excluded from
        the fit, so a minority of stray/noise peaks can't drag the curve.
    sigma : float
        Outlier threshold, in robust standard deviations. Larger keeps more
        points; smaller is more aggressive.
    """
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float)
    good = np.isfinite(H) & np.isfinite(f)
    H, f = H[good], f[good]
    n_total = int(H.size)

    n_params = len(model.param_names)
    if min_points is None:
        min_points = max(3, n_params + 1)

    nan_p = np.full(n_params, np.nan)

    if n_total < min_points:
        return PeakFit(peak_index, model.key, nan_p.copy(), nan_p.copy(), np.nan,
                       n_total, n_total=n_total, success=False,
                       message=f"need >= {min_points} points to fit, got {n_total}")

    # --- Outlier rejection: keep only the majority-trend points ---
    # Stage 1: an f^2-space RANSAC consensus for the initial inlier set.
    if reject_outliers:
        mask = _robust_inlier_mask(H, f, sigma=sigma)
    else:
        mask = np.ones(n_total, dtype=bool)

    if mask.sum() < min_points:
        mask = np.ones(n_total, dtype=bool)  # too aggressive; keep all

    floor = max(min_points, int(0.3 * n_total))

    # Stage 2: iterate a robust (Cauchy) model fit and re-clip in *model*
    # space. This removes high-leverage points near the majority's edge that
    # the f^2 stage can miss, and converges the mask to the majority trend.
    if reject_outliers and _HAS_SCIPY:
        for _ in range(6):
            try:
                p0 = _linearized_estimate(H[mask], f[mask], model.key)
            except Exception:
                break
            params_rob, _ = _robust_model_fit(model.func, H[mask], f[mask], p0)
            resid = f - model.func(H, *params_rob)
            med = np.median(resid[mask])
            s = _robust_scale(resid[mask])
            new_mask = np.abs(resid - med) < sigma * s
            if new_mask.sum() < floor:
                order = np.argsort(np.abs(resid - med))
                new_mask = np.zeros(n_total, dtype=bool)
                new_mask[order[:floor]] = True
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

    H_in, f_in = H[mask], f[mask]
    H_out, f_out = H[~mask], f[~mask]
    n_in = int(H_in.size)

    if n_in < min_points:
        # Rejection was too aggressive for this track; fall back to all points.
        H_in, f_in = H, f
        H_out, f_out = H[:0], f[:0]
        n_in = n_total

    try:
        p0 = _linearized_estimate(H_in, f_in, model.key)
    except Exception as exc:
        return PeakFit(peak_index, model.key, nan_p.copy(), nan_p.copy(), np.nan,
                       n_in, n_total=n_total, n_rejected=int(H_out.size),
                       H_out=H_out, f_out=f_out, success=False,
                       message=f"linearized estimate failed ({exc})")

    params = np.asarray(p0, dtype=float)
    perr = nan_p.copy()

    # Stage 3: a clean ordinary fit on the retained inliers for final
    # parameters and proper standard errors; if it fails, keep a robust fit.
    if _HAS_SCIPY:
        try:
            popt, pcov = curve_fit(model.func, H_in, f_in, p0=p0, maxfev=20000)
            params = np.asarray(popt, dtype=float)
            with np.errstate(invalid="ignore"):
                perr = np.sqrt(np.abs(np.diag(pcov)))
        except Exception:
            params, perr = _robust_model_fit(model.func, H_in, f_in, p0)

    # R^2 computed on the inliers (the trend we actually fit).
    f_pred = model.func(H_in, *params)
    ss_res = float(np.sum((f_in - f_pred) ** 2))
    ss_tot = float(np.sum((f_in - np.mean(f_in)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    derived = {}
    if model.derived is not None:
        try:
            derived = model.derived(params)
        except Exception:
            derived = {}

    # Draw the smooth curve only across the inlier field range, so it traces
    # the majority trend rather than extrapolating over the outlier region.
    Hs = np.linspace(float(np.min(H_in)), float(np.max(H_in)), n_curve)
    fs = model.func(Hs, *params)

    return PeakFit(peak_index, model.key, params, np.asarray(perr, dtype=float),
                   r2, n_in, Hs, fs, derived,
                   n_total=n_total, n_rejected=int(H_out.size),
                   H_out=H_out, f_out=f_out, success=True)


def fit_peak_result(peak_result, model_key: str = DEFAULT_MODEL_KEY,
                    n_curve: int = 200, reject_outliers: bool = True,
                    sigma: float = 3.0) -> DatasetFit:
    """Fit every peak track in a :class:`peak_analysis.PeakResult`.

    ``peak_result`` is duck-typed: needs ``.label``, ``.num_peaks`` and
    ``.track(i) -> (frequencies, fields)``. ``reject_outliers`` / ``sigma``
    control majority-trend outlier rejection (see :func:`fit_track`).
    """
    model = FIT_MODELS_BY_KEY[model_key]
    dataset_fit = DatasetFit(label=peak_result.label, model=model)

    for i in range(peak_result.num_peaks):
        freqs, fields = peak_result.track(i)   # (frequencies, field_positions)
        # Fit frequency as a function of field: x = field, y = frequency.
        dataset_fit.peak_fits.append(
            fit_track(fields, freqs, model, peak_index=i, n_curve=n_curve,
                      reject_outliers=reject_outliers, sigma=sigma)
        )

    return dataset_fit


def format_fit(model: FitModel, pf: PeakFit) -> str:
    """One-line human-readable summary of a peak fit."""
    head = f"Peak {pf.peak_index + 1}: "
    if not pf.success:
        return head + f"fit failed - {pf.message}"

    parts = []
    for name, val, err in zip(model.param_names, pf.params, pf.param_errors):
        if np.isfinite(err):
            parts.append(f"{name} = {val:.5g} +/- {err:.2g}")
        else:
            parts.append(f"{name} = {val:.5g}")
    r2 = f"{pf.r_squared:.4f}" if np.isfinite(pf.r_squared) else "n/a"
    pts = f"n = {pf.n_points}"
    if pf.n_rejected:
        pts += f" (rejected {pf.n_rejected} of {pf.n_total} as outliers)"
    line = head + ", ".join(parts) + f"  (R^2 = {r2}, {pts})"

    if pf.derived:
        dparts = [f"{k} = {v:.5g}" for k, v in pf.derived.items()]
        line += "\n      derived: " + ", ".join(dparts)
    return line
