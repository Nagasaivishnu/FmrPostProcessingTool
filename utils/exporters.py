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


def save_figure(figure, output_path: str, fmt: str = "png", dpi: int = 300) -> None:
    """Save a Matplotlib figure to disk in the requested format/DPI.

    fmt : "png" | "svg" | "pdf" | "eps"
    """
    output_path = Path(output_path).with_suffix(f".{fmt.lower()}")
    figure.savefig(output_path, format=fmt.lower(), dpi=dpi, bbox_inches="tight")
