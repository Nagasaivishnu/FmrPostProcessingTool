"""
processing/loader.py

Handles discovery and loading of broadband FMR CSV files from a directory.
Each file in a dataset directory corresponds to a single microwave frequency
and contains a magnetic-field sweep with voltageX / voltageY (or signal)
columns.

This module is intentionally free of any Qt dependency so it can be reused,
unit tested, or scripted outside of the GUI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Filenames are expected to contain something like "5.0GHz" or "mmWave5.0GHz".
# We try a couple of common patterns and fall back to "any number before GHz".
_FREQ_PATTERNS = [
    re.compile(r'mmWave([\d.]+)\s*GHz', re.IGNORECASE),
    re.compile(r'([\d.]+)\s*GHz', re.IGNORECASE),
]

# Minimum number of points for a file to count as a real field sweep.
#
# Aborted or interrupted acquisitions leave behind stub files holding only the
# first few rows written before the sweep actually started - typically three
# rows all sitting at the starting field. These are not sweeps: they carry no
# lineshape, and letting them through poisons every dataset-wide statistic
# downstream. Most importantly they corrupt the common field grid used to
# build the 2D heatmap, because that grid is sized and bounded by statistics
# taken over *all* records.
MIN_SWEEP_POINTS = 10


def extract_frequency(filename: str) -> Optional[float]:
    """Extract the microwave frequency (in GHz) encoded in a filename.

    Tries a small set of known patterns (e.g. ``FMR_mmWave5.0GHz.csv``,
    ``5.25GHz_scan.csv``). Returns ``None`` if no pattern matches.
    """
    for pattern in _FREQ_PATTERNS:
        match = pattern.search(filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


@dataclass
class FrequencyRecord:
    """Raw data extracted from a single FMR file (one frequency)."""

    frequency: float
    H_field: np.ndarray
    voltage_x: np.ndarray
    voltage_y: np.ndarray
    filename: str
    bg_voltage_x: Optional[np.ndarray] = None
    bg_voltage_y: Optional[np.ndarray] = None
    bg_filename: Optional[str] = None


@dataclass
class Dataset:
    """A full experimental (or background) dataset: many frequencies."""

    label: str
    directory: Path
    records: Dict[float, FrequencyRecord] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def sorted_frequencies(self) -> List[float]:
        return sorted(self.records.keys())

    @property
    def n_files(self) -> int:
        return len(self.records)

    @property
    def freq_range(self):
        if not self.records:
            return None
        freqs = self.sorted_frequencies
        return (freqs[0], freqs[-1])

    @property
    def field_range(self):
        if not self.records:
            return None
        first = self.records[self.sorted_frequencies[0]]
        return (float(np.min(first.H_field)), float(np.max(first.H_field)))

    def summary_text(self) -> str:
        if not self.records:
            return "No data loaded."
        fmin, fmax = self.freq_range
        hmin, hmax = self.field_range
        lines = [
            f"Files loaded: {self.n_files}",
            f"Frequency range: {fmin:g} - {fmax:g} GHz",
            f"Field range: {hmin:.6g} - {hmax:.6g} (native units)",
        ]
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        return "\n".join(lines)


def _find_fmr_files(directory: Path) -> List[Path]:
    """Find candidate FMR data files in a directory (recursive).

    Accepts .csv and .txt files that contain 'FMR' in the name, or any
    .csv if none match that stricter pattern (graceful fallback).
    """
    strict = sorted(directory.glob("**/*FMR*.csv")) + sorted(directory.glob("**/*FMR*.txt"))
    if strict:
        return strict
    # Fallback: any CSV in the directory (last resort, still recursive).
    fallback = sorted(directory.glob("**/*.csv"))
    return fallback


def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/TXT file into a DataFrame, tolerating both comma and
    whitespace/tab delimited text files.
    """
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, delim_whitespace=True)


def _extract_columns(df: pd.DataFrame, warnings: List[str], filename: str):
    """Pull field / voltageX / voltageY columns out of a dataframe,
    tolerating a few naming variants and missing voltageY (single-channel
    signal).
    """
    cols = {c.lower(): c for c in df.columns}

    def find(*candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    field_col = find("field", "h_field", "hfield", "magnetic field", "h")
    x_col = find("voltagex", "voltage_x", "vx", "signalx", "lock-in x")
    y_col = find("voltagey", "voltage_y", "vy", "signaly", "lock-in y")

    if field_col is None:
        raise ValueError(f"Could not find a 'field' column in {filename}")
    if x_col is None:
        raise ValueError(f"Could not find a 'voltageX' column in {filename}")

    H_field = np.asarray(df[field_col], dtype=float)
    volx = np.asarray(df[x_col], dtype=float)

    if y_col is not None:
        voly = np.asarray(df[y_col], dtype=float)
    else:
        warnings.append(f"{filename}: no voltageY column found, using zeros")
        voly = np.zeros_like(volx)

    return H_field, volx, voly


def load_dataset(directory: str, label: Optional[str] = None,
                 min_points: int = MIN_SWEEP_POINTS) -> Dataset:
    """Load every FMR file in ``directory`` into a :class:`Dataset`.

    Parameters
    ----------
    directory : str or Path
        Folder containing one file per microwave frequency.
    label : str, optional
        Human readable label used in legends/exports. Defaults to the
        directory's folder name.
    min_points : int, optional
        Files with fewer than this many rows are treated as aborted
        acquisitions and skipped (see :data:`MIN_SWEEP_POINTS`). Pass ``0``
        to disable the check and load everything.
    """
    directory = Path(directory)
    label = label or directory.name
    dataset = Dataset(label=label, directory=directory)
    skipped_short: List[tuple] = []

    if not directory.exists():
        dataset.warnings.append(f"Directory does not exist: {directory}")
        return dataset

    files = _find_fmr_files(directory)
    if not files:
        dataset.warnings.append(f"No FMR files found in {directory}")
        return dataset

    for fpath in files:
        freq = extract_frequency(fpath.name)
        if freq is None:
            dataset.warnings.append(f"Skipped (no frequency in filename): {fpath.name}")
            continue
        try:
            df = _read_table(fpath)
            H_field, volx, voly = _extract_columns(df, dataset.warnings, fpath.name)
        except Exception as exc:
            dataset.warnings.append(f"Failed to read {fpath.name}: {exc}")
            continue

        if min_points and len(H_field) < min_points:
            skipped_short.append((freq, fpath.name, len(H_field)))
            continue

        if freq in dataset.records:
            dataset.warnings.append(
                f"Duplicate frequency {freq:g} GHz - keeping first file, "
                f"ignoring {fpath.name}"
            )
            continue

        dataset.records[freq] = FrequencyRecord(
            frequency=freq,
            H_field=H_field,
            voltage_x=volx,
            voltage_y=voly,
            filename=fpath.name,
        )

    if skipped_short:
        # Aggregate into a single warning: an aborted run can leave hundreds
        # of stubs behind, and one line each would bury every other message.
        freqs = sorted(f for f, _n, _l in skipped_short)
        rows = sorted({l for _f, _n, l in skipped_short})
        dataset.warnings.append(
            f"Skipped {len(skipped_short)} file(s) with fewer than {min_points} "
            f"points (aborted sweeps, {rows[0]}-{rows[-1]} rows each), "
            f"covering {freqs[0]:g}-{freqs[-1]:g} GHz. "
            f"First: {skipped_short[0][1]}"
        )

    return dataset


def attach_background(dataset: Dataset, background: Dataset) -> None:
    """Attach matching-frequency background traces onto ``dataset`` records.

    Matching is done by exact frequency, falling back to the nearest
    available background frequency if there's no exact match. The
    background field axis does not need to be the same length as the
    experimental axis - interpolation happens later, during preprocessing.
    """
    if not background.records:
        dataset.warnings.append("Background dataset is empty - skipping subtraction setup")
        return

    bg_freqs = np.array(background.sorted_frequencies)

    for freq, record in dataset.records.items():
        if freq in background.records:
            bg_record = background.records[freq]
        else:
            nearest_idx = int(np.argmin(np.abs(bg_freqs - freq)))
            nearest_freq = bg_freqs[nearest_idx]
            bg_record = background.records[nearest_freq]
            dataset.warnings.append(
                f"No exact background match for {freq:g} GHz, "
                f"using nearest background at {nearest_freq:g} GHz"
            )
        record.bg_voltage_x = bg_record.voltage_x
        record.bg_voltage_y = bg_record.voltage_y
        record.bg_filename = bg_record.filename
