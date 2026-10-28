"""
plotting/heatmap_render.py

One place that draws an FMR heatmap onto a Matplotlib Figure, so the
Heatmap tab (on screen), "Save All", and the Export tab all produce the
same picture from the same display settings.

Default look is a publication-ready, Origin-style figure:
  - Arial (or a metric-compatible fallback), regular weight
  - closed black frame, ticks pointing inward on all four sides,
    minor ticks, line widths scaled to the font size
  - colour bar exactly as tall as the plot, inward ticks, and any
    "1e-10" multiplier moved into the colour-bar label as (x10^-10)
  - exact physical figure width (e.g. 8.5 cm single journal column)
  - vector export (PDF / SVG / EPS) with real, editable text

HeatmapStyle holds every display setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

# Aspect presets: name -> (width, height) of the plot area. None = fill window.
ASPECT_PRESETS = {
    "4:3": (4.0, 3.0),
    "1:1 (square)": (1.0, 1.0),
    "3:2": (3.0, 2.0),
    "16:9": (16.0, 9.0),
    "3:4 (tall)": (3.0, 4.0),
    "Auto (fill window)": None,
    "Custom": "custom",
}

# Common journal figure widths (cm).
WIDTH_PRESETS_CM = {
    "8.5 cm (single column)": 8.5,
    "11.4 cm (1.5 column)": 11.4,
    "17.5 cm (double column)": 17.5,
    "Custom": None,
}

SAVE_FORMATS = ["png", "tiff", "pdf", "svg", "eps"]

# Origin uses Arial; fall back to metric-compatible look-alikes.
ORIGIN_FONTS = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
_origin_font_cache: Optional[str] = None

CM_PER_INCH = 2.54
PT_PER_INCH = 72.0


def origin_font() -> str:
    """First installed font from ORIGIN_FONTS (looked up once), so
    Matplotlib doesn't log a 'font not found' warning for every label.
    """
    global _origin_font_cache
    if _origin_font_cache is None:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        _origin_font_cache = next((f for f in ORIGIN_FONTS if f in installed),
                                  "sans-serif")
    return _origin_font_cache


@dataclass
class HeatmapStyle:
    # Colour scaling / sampling
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    interpolation: str = "nearest"
    normalize: bool = False          # divide by max |intensity| -> 0..1
    cbar_power_in_label: bool = True  # "Intensity (x10^-10)" instead of "1e-10"

    # Text
    font_size: float = 9.0
    bold_text: bool = False
    show_title: bool = False
    x_label: str = "Magnetic Field (T)"
    y_label: str = "Frequency (GHz)"
    cbar_label: str = "Intensity (a.u.)"

    # Look
    origin_style: bool = True

    # Plot-area aspect ratio (width, height); None = fill the window/figure.
    aspect: Optional[Tuple[float, float]] = (4.0, 3.0)

    # Output
    width_cm: float = 8.5
    dpi: int = 600
    save_format: str = "png"
    wysiwyg: bool = True  # on-screen preview scaled to the saved figure

    # Two-colour overlay of two datasets (additive colour mixing)
    overlay_pair: str = "Magenta / Green"
    overlay_background: str = "Black"        # "Black" (additive) | "White"
    overlay_signal: str = "Absolute value"   # how negative values are treated
    overlay_saturation: float = 99.5         # percentile mapped to full colour
    gain_a: float = 1.0
    gain_b: float = 1.0

    @property
    def width_in(self) -> float:
        return self.width_cm / CM_PER_INCH

    def png_figsize(self, screen_figsize: Optional[Tuple[float, float]] = None):
        """Approximate saved-figure size (inches). The exact height for a
        fixed aspect ratio is refined in :func:`fit_figure_height`.
        """
        w = self.width_in
        if self.aspect is not None:
            aw, ah = self.aspect
            fs_in = self.font_size / PT_PER_INCH
            plot_w = max(w - 9.5 * fs_in - 0.06 * w, 0.3 * w)
            return w, plot_w * ah / aw + 4.0 * fs_in
        if screen_figsize is not None and screen_figsize[0] > 0:
            return w, w * screen_figsize[1] / screen_figsize[0]
        return w, w * 0.75


# --------------------------------------------------------------- drawing

def render_heatmap(fig, result, label: str, cmap: str, style: HeatmapStyle,
                   apply_ranges: Optional[Callable] = None, scale: float = 1.0):
    """Clear ``fig`` and draw ``result``'s heatmap onto it. Returns the Axes.

    ``apply_ranges`` (optional) is called with the Axes to apply the global
    display ranges. ``scale`` multiplies every font size / line width (used
    for the on-screen preview, so it looks like the saved figure).
    """
    fig.clear()
    try:
        fig.set_layout_engine("constrained", w_pad=2 / 72, h_pad=2 / 72)
    except Exception:  # older Matplotlib
        pass
    ax = fig.add_subplot(111)

    if result is None or not result.sorted_frequencies:
        ax.text(0.5, 0.5, "No processed data.\nRun 'Process All Datasets' in Tab 2.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    fs = style.font_size * scale
    origin = style.origin_style
    family = origin_font() if origin else None
    weight = "bold" if style.bold_text else "normal"
    text_kw = {"fontsize": fs, "fontweight": weight}
    if family:
        text_kw["fontfamily"] = family

    field_grid, freqs, matrix = result.as_matrix()
    if style.normalize:
        peak = np.nanmax(np.abs(matrix))
        if peak > 0:
            matrix = matrix / peak

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation=style.interpolation,
        extent=[field_grid.min(), field_grid.max(), freqs.min(), freqs.max()],
        vmin=style.vmin,
        vmax=style.vmax,
        rasterized=True,  # keep PDF/SVG small; text and axes stay vector
    )

    ax.set_xlabel(style.x_label, **text_kw)
    ax.set_ylabel(style.y_label, **text_kw)
    if style.show_title:
        ax.set_title(label, **dict(text_kw, fontsize=fs * 1.1))

    if style.aspect is not None:
        aw, ah = style.aspect
        ax.set_box_aspect(ah / aw)

    # Colour bar attached to the plot: always exactly as tall as it.
    cax = ax.inset_axes([1.035, 0.0, 0.045, 1.0])
    cbar = fig.colorbar(im, cax=cax)

    cbar_label = style.cbar_label
    if style.normalize and cbar_label.strip().lower() in ("intensity", "intensity (a.u.)"):
        cbar_label = "Normalized intensity"
    if style.cbar_power_in_label and not style.normalize:
        power = _common_power(im.norm.vmin, im.norm.vmax)
        if power:
            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _pos, p=power: _fmt_tick(v / 10.0 ** p)))
            cbar_label = f"{cbar_label} (×10$^{{{power}}}$)"
    cbar.set_label(cbar_label, **text_kw)
    offset = cbar.ax.yaxis.get_offset_text()
    offset.set_fontsize(fs)
    offset.set_fontweight(weight)
    if family:
        offset.set_fontfamily(family)

    _style_axis(ax, fs, family, weight, origin)
    _style_axis(cbar.ax, fs, family, weight, origin, is_cbar=True)
    cbar.outline.set_linewidth(_frame_lw(fs) if origin else 0.8 * scale)

    if apply_ranges is not None:
        apply_ranges(ax)
    return ax


def _common_power(vmin, vmax) -> int:
    """Power of ten to pull out of the colour-bar ticks (0 = none needed)."""
    big = max(abs(vmin or 0.0), abs(vmax or 0.0))
    if not np.isfinite(big) or big == 0:
        return 0
    p = int(math.floor(math.log10(big)))
    return p if (p <= -3 or p >= 4) else 0


def _fmt_tick(v: float) -> str:
    s = f"{v:.3g}"
    return s.replace("-", "−")  # proper minus sign


def _frame_lw(fs: float) -> float:
    return max(0.6, fs / 9.0)  # ~1 pt frame at 9 pt text


def _set_tick_font(ax, family, weight):
    """Font family/weight on tick labels. Matplotlib copies the first
    tick's text properties to ticks it creates later (zoom/pan).
    """
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis.get_major_ticks():
            for lbl in (tick.label1, tick.label2):
                if family:
                    lbl.set_fontfamily(family)
                lbl.set_fontweight(weight)


def _style_axis(ax, fs, family, weight, origin, is_cbar=False):
    if not origin:
        ax.tick_params(labelsize=fs)
        _set_tick_font(ax, family, weight)
        return

    lw = _frame_lw(fs)
    major_len, minor_len = 0.45 * fs, 0.25 * fs
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
        spine.set_color("black")

    common = dict(direction="in", colors="black", labelcolor="black")
    pad = 0.35 * fs
    if is_cbar:
        ax.tick_params(which="major", length=major_len * 0.8, width=lw * 0.9,
                       right=True, left=False, labelsize=fs, pad=pad, **common)
        ax.tick_params(which="minor", length=minor_len * 0.8, width=lw * 0.7,
                       right=True, left=False, **common)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    else:
        ax.tick_params(which="major", length=major_len, width=lw * 0.9,
                       top=True, right=True, labelsize=fs, pad=pad, **common)
        ax.tick_params(which="minor", length=minor_len, width=lw * 0.7,
                       top=True, right=True, **common)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    _set_tick_font(ax, family, weight)


# ---------------------------------------------------------------- saving

def fit_figure_height(fig, style: HeatmapStyle) -> None:
    """For a fixed aspect ratio, adjust the figure height (width stays
    exact) so there is no empty band above/below or beside the plot.
    """
    if style.aspect is None:
        return
    w, h = fig.get_size_inches()
    for _ in range(4):
        fig.canvas.draw()  # also runs the constrained layout
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        slack_w, slack_h = w - tb.width, h - tb.height
        if slack_w > 0.04 * w:        # plot limited by height -> make taller
            h *= w / max(tb.width, 1e-6)
        elif slack_h > 0.02 * h:      # spare vertical space -> make shorter
            h = tb.height + 0.01 * h
        else:
            break
        fig.set_size_inches(w, h)


def save_heatmap_figure(fig, path: str, fmt: str, dpi: int) -> None:
    """Save at the figure's exact size. Text stays real (editable) text in
    PDF / EPS / SVG; raster formats get a white background."""
    fmt = fmt.lower()
    rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    kwargs = {"format": fmt, "dpi": dpi, "facecolor": "white"}
    if fmt in ("tif", "tiff"):
        kwargs["format"] = "tiff"
        kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    with matplotlib.rc_context(rc):
        fig.savefig(path, **kwargs)


# ------------------------------------------------------ two-colour overlay

# Colour pairs whose sum is white, so overlapping modes show as white on a
# black background (or black on a white background).
OVERLAY_PAIRS = {
    "Magenta / Green": ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "Red / Cyan": ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
    "Orange / Blue (colour-blind safer)": ((1.0, 0.6, 0.0), (0.0, 0.4, 1.0)),
}
OVERLAY_BACKGROUNDS = ["Black", "White"]
OVERLAY_SIGNALS = ["Absolute value", "Positive part only", "Negative part only"]


def _common_grid(res_a, res_b):
    """Field and frequency axes covering both datasets."""
    fa, qa, _ = res_a.as_matrix()
    fb, qb, _ = res_b.as_matrix()
    lo, hi = min(fa.min(), fb.min()), max(fa.max(), fb.max())
    step = min(np.median(np.diff(np.sort(fa))) if fa.size > 1 else np.inf,
               np.median(np.diff(np.sort(fb))) if fb.size > 1 else np.inf)
    n = int(np.clip((hi - lo) / step + 1 if np.isfinite(step) and step > 0 else 500,
                    50, 4000))
    field = np.linspace(lo, hi, n)
    freqs = np.unique(np.concatenate([qa, qb]))
    return field, freqs


def _on_grid(result, field, freqs):
    """Resample a dataset onto (freqs x field); zero where it has no data."""
    f_own, q_own, m_own = result.as_matrix()
    order = np.argsort(f_own)
    f_own, m_own = f_own[order], m_own[:, order]
    q_step = np.median(np.diff(q_own)) if q_own.size > 1 else np.inf
    out = np.zeros((freqs.size, field.size))
    for i, q in enumerate(freqs):
        j = int(np.argmin(np.abs(q_own - q)))
        if abs(q_own[j] - q) > 0.51 * q_step:
            continue  # this frequency wasn't measured for this dataset
        out[i] = np.interp(field, f_own, m_own[j], left=0.0, right=0.0)
    return out


def _channel(matrix, style, gain):
    """Map a dataset to 0..1 brightness for its colour channel.

    Each frequency row's baseline (its median, since resonance lines are
    narrow) is removed first, so only the modes light up, not offsets.
    """
    matrix = np.array(matrix, dtype=float)
    has_data = np.any(matrix != 0, axis=1)
    if np.any(has_data):
        matrix[has_data] -= np.median(matrix[has_data], axis=1, keepdims=True)
    if style.overlay_signal == "Positive part only":
        m = np.clip(matrix, 0, None)
    elif style.overlay_signal == "Negative part only":
        m = np.clip(-matrix, 0, None)
    else:
        m = np.abs(matrix)
    nz = m[m > 0]
    top = np.percentile(nz, style.overlay_saturation) if nz.size else 1.0
    if not top > 0:
        top = 1.0
    return np.clip(m / top * gain, 0.0, 1.0)


def render_overlay(fig, res_a, label_a, res_b, label_b, style: HeatmapStyle,
                   apply_ranges: Optional[Callable] = None, scale: float = 1.0):
    """Draw datasets A and B in one plot as two colour channels. Where both
    have signal the colours add (white on black / black on white)."""
    fig.clear()
    try:
        fig.set_layout_engine("constrained", w_pad=2 / 72, h_pad=2 / 72)
    except Exception:
        pass
    ax = fig.add_subplot(111)

    if (res_a is None or res_b is None or not res_a.sorted_frequencies
            or not res_b.sorted_frequencies):
        ax.text(0.5, 0.5, "Pick two processed datasets (A and B).",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    fs = style.font_size * scale
    origin = style.origin_style
    family = origin_font() if origin else None
    weight = "bold" if style.bold_text else "normal"
    text_kw = {"fontsize": fs, "fontweight": weight}
    if family:
        text_kw["fontfamily"] = family

    field, freqs = _common_grid(res_a, res_b)
    na = _channel(_on_grid(res_a, field, freqs), style, style.gain_a)
    nb = _channel(_on_grid(res_b, field, freqs), style, style.gain_b)
    col_a, col_b = (np.array(c) for c in OVERLAY_PAIRS.get(
        style.overlay_pair, next(iter(OVERLAY_PAIRS.values()))))

    white_bg = style.overlay_background == "White"
    if white_bg:
        rgb = 1.0 - np.clip(na[..., None] * (1 - col_a) + nb[..., None] * (1 - col_b), 0, 1)
        both = (0.0, 0.0, 0.0)
    else:
        rgb = np.clip(na[..., None] * col_a + nb[..., None] * col_b, 0, 1)
        both = (1.0, 1.0, 1.0)

    ax.imshow(rgb, aspect="auto", origin="lower", interpolation=style.interpolation,
              extent=[field.min(), field.max(), freqs.min(), freqs.max()],
              rasterized=True)

    ax.set_xlabel(style.x_label, **text_kw)
    ax.set_ylabel(style.y_label, **text_kw)
    if style.show_title:
        ax.set_title(f"{label_a} vs {label_b}", **dict(text_kw, fontsize=fs * 1.1))
    if style.aspect is not None:
        aw, ah = style.aspect
        ax.set_box_aspect(ah / aw)

    # Key above the plot: colour A, colour B, overlap.
    from matplotlib.patches import Patch
    edge = "0.3"
    handles = [Patch(facecolor=tuple(col_a), edgecolor=edge, label=label_a),
               Patch(facecolor=tuple(col_b), edgecolor=edge, label=label_b),
               Patch(facecolor=both, edgecolor=edge, label="both")]
    leg_kw = {"size": fs, "weight": weight}
    if family:
        leg_kw["family"] = family
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=3, frameon=False, prop=leg_kw, handlelength=1.2,
              handleheight=0.9, columnspacing=1.2, borderaxespad=0.1)

    _style_axis(ax, fs, family, weight, origin)
    if not white_bg and origin:
        # inward ticks must stay visible on the dark image
        ax.tick_params(which="both", color="white")
        for spine in ax.spines.values():
            spine.set_color("black")

    if apply_ranges is not None:
        apply_ranges(ax)
    return ax


def configure_mathtext_font() -> None:
    """Make the (x10^n) superscript use the same font as the labels."""
    family = origin_font()
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = family
    matplotlib.rcParams["mathtext.it"] = f"{family}:italic"
    matplotlib.rcParams["mathtext.bf"] = f"{family}:bold"
    matplotlib.rcParams["mathtext.sf"] = family
    matplotlib.rcParams["mathtext.cal"] = family  # avoid 'cursive' lookup warning


configure_mathtext_font()
