"""
utils/exporters.py

Export helpers for processed FMR data: full heatmap matrices, 1D slices
(frequency-cut / field-cut), and saved figures. Kept Qt-free so they can be
called from the GUI's export tab or scripted directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from processing.dataset_processor import ProcessedDataset
from processing.peak_analysis import PeakResult


def export_heatmap(result: ProcessedDataset, output_path: str, fmt: str = "csv") -> None:
    """Export a processed dataset's full 2D matrix (field, frequency,
    intensity) to disk.

    fmt : "csv" | "txt" | "npy" | "mat"
    """
    output_path = Path(output_path)
    field_grid, freqs, matrix = result.as_matrix()

    fmt = fmt.lower()
    if fmt == "npy":
        np.savez(output_path.with_suffix(".npz"), field=field_grid, frequency=freqs, intensity=matrix)
    elif fmt == "mat":
        try:
            from scipy.io import savemat
        except ImportError as exc:
            raise RuntimeError("scipy is required to export .mat files") from exc
        savemat(output_path.with_suffix(".mat"),
                {"field": field_grid, "frequency": freqs, "intensity": matrix})
    elif fmt in ("csv", "txt"):
        sep = "," if fmt == "csv" else "\t"
        data = {"field": field_grid}
        for i, freq in enumerate(freqs):
            data[f"intensity_{freq:g}GHz"] = matrix[i, :]
        df = pd.DataFrame(data)
        df.to_csv(output_path, sep=sep, index=False)
    else:
        raise ValueError(f"Unsupported heatmap export format: {fmt}")


def export_slice(x_values: np.ndarray, series: Dict[str, np.ndarray],
                  output_path: str, fmt: str = "csv",
                  x_label: str = "x") -> None:
    """Export one or more 1D slices that share a common x-axis.

    Parameters
    ----------
    x_values : array
        Shared independent-axis values (field or frequency).
    series : dict
        Mapping of {dataset_label: y_values}, one column per dataset.
    fmt : "csv" | "txt" | "excel"
    """
    output_path = Path(output_path)
    data = {x_label: x_values}
    data.update(series)
    df = pd.DataFrame(data)

    fmt = fmt.lower()
    if fmt == "excel":
        df.to_excel(output_path.with_suffix(".xlsx"), index=False)
    elif fmt == "txt":
        df.to_csv(output_path, sep="\t", index=False)
    else:
        df.to_csv(output_path, index=False)


def export_peak_data(peak_results: Dict[str, PeakResult], output_path: str,
                      fmt: str = "csv") -> None:
    """Export peak-tracking results (one or more datasets) to a flat,
    long-format table: one row per (dataset, frequency, peak index).

    A long format is used (rather than one column per peak) because
    different datasets can have different frequency axes, so there's no
    single shared x-axis to align wide columns against - the same
    convention problem that motivated keeping this separate from
    :func:`export_slice`.

    Parameters
    ----------
    peak_results : dict
        Mapping of {dataset_label: PeakResult}.
    fmt : "csv" | "txt" | "excel"
    """
    output_path = Path(output_path)

    rows = []
    for label, peak_result in peak_results.items():
        for peak_idx in range(peak_result.num_peaks):
            for freq, peak_field, peak_height in zip(
                peak_result.frequencies,
                peak_result.peak_fields[peak_idx, :],
                peak_result.peak_heights[peak_idx, :],
            ):
                rows.append({
                    "dataset": label,
                    "peak_index": peak_idx + 1,
                    "frequency_GHz": freq,
                    "field": peak_field,
                    "intensity": peak_height,
                })

    df = pd.DataFrame(rows, columns=["dataset", "peak_index", "frequency_GHz", "field", "intensity"])

    fmt = fmt.lower()
    if fmt == "excel":
        df.to_excel(output_path.with_suffix(".xlsx"), index=False)
    elif fmt == "txt":
        df.to_csv(output_path, sep="\t", index=False)
    else:
        df.to_csv(output_path, index=False)


def export_fit_parameters(dataset_fits: Dict[str, "object"], output_path: str,
                          fmt: str = "csv") -> None:
    """Export curve-fit parameters to a flat table: one row per
    (dataset, peak). Columns include the model, formula, each fitted
    constant with its standard error, R-squared, point count, and success
    flag.

    ``dataset_fits`` maps {dataset_label: DatasetFit}; it is duck-typed
    (only ``.model`` and ``.peak_fits`` are accessed) to keep this module
    free of a hard dependency on :mod:`processing.curve_fitting`.

    fmt : "csv" | "txt" | "excel"
    """
    output_path = Path(output_path)

    rows = []
    for label, dfit in dataset_fits.items():
        model = dfit.model
        for pf in dfit.peak_fits:
            row = {
                "dataset": label,
                "peak_index": pf.peak_index + 1,
                "model": model.key,
                "formula": model.formula,
                "n_points": pf.n_points,
                "n_rejected": getattr(pf, "n_rejected", 0),
                "n_total": getattr(pf, "n_total", pf.n_points),
                "r_squared": pf.r_squared,
                "success": pf.success,
            }
            for name, val, err in zip(model.param_names, pf.params, pf.param_errors):
                row[name] = val
                row[f"{name}_err"] = err
            for dname, dval in getattr(pf, "derived", {}).items():
                row[f"derived_{dname}"] = dval
            rows.append(row)

    df = pd.DataFrame(rows)

    fmt = fmt.lower()
    if fmt == "excel":
        df.to_excel(output_path.with_suffix(".xlsx"), index=False)
    elif fmt == "txt":
        df.to_csv(output_path, sep="\t", index=False)
    else:
        df.to_csv(output_path, index=False)


def save_figure(figure, output_path: str, fmt: str = "png", dpi: int = 300) -> None:
    """Save a Matplotlib figure to disk in the requested format/DPI.

    fmt : "png" | "svg" | "pdf" | "eps"
    """
    output_path = Path(output_path).with_suffix(f".{fmt.lower()}")
    figure.savefig(output_path, format=fmt.lower(), dpi=dpi, bbox_inches="tight")
