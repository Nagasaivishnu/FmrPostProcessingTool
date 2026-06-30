"""
processing/dataset_processor.py

Ties together loader.Dataset + preprocessing.PreprocessSettings to produce
a fully processed 2D dataset (field x frequency x intensity), ready for
heatmap plotting, slicing, and export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .loader import Dataset
from .preprocessing import PreprocessSettings, preprocess_trace


@dataclass
class ProcessedDataset:
    """Result of running the preprocessing pipeline over an entire
    experimental Dataset.

    ``processed`` maps frequency -> 1D processed trace (same length as the
    corresponding H_field array for that frequency).
    """

    label: str
    sorted_frequencies: List[float] = field(default_factory=list)
    H_field_by_freq: Dict[float, np.ndarray] = field(default_factory=dict)
    processed: Dict[float, np.ndarray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def common_field_grid(self, n_points: Optional[int] = None) -> np.ndarray:
        """Build a common field axis spanning the intersection of all
        per-frequency field ranges, used for the 2D heatmap matrix.
        """
        if not self.sorted_frequencies:
            return np.array([])

        mins, maxs, lengths = [], [], []
        for freq in self.sorted_frequencies:
            H = self.H_field_by_freq[freq]
            mins.append(np.min(H))
            maxs.append(np.max(H))
            lengths.append(len(H))

        common_min = max(mins)
        common_max = min(maxs)
        if n_points is None:
            n_points = int(np.median(lengths))
        if common_min >= common_max:
            # Field ranges don't overlap meaningfully - fall back to the
            # full span of the first trace rather than crashing.
            first = self.H_field_by_freq[self.sorted_frequencies[0]]
            return np.linspace(np.min(first), np.max(first), n_points)
        return np.linspace(common_min, common_max, n_points)

    def as_matrix(self, field_grid: Optional[np.ndarray] = None):
        """Interpolate every frequency's trace onto a common field grid and
        stack into a 2D matrix of shape (n_frequencies, n_field_points).

        Returns (field_grid, freq_array, matrix).
        """
        if field_grid is None:
            field_grid = self.common_field_grid()

        matrix = np.zeros((len(self.sorted_frequencies), len(field_grid)))
        for i, freq in enumerate(self.sorted_frequencies):
            H = self.H_field_by_freq[freq]
            sig = self.processed[freq]
            order = np.argsort(H)
            matrix[i, :] = np.interp(field_grid, H[order], sig[order])

        return field_grid, np.array(self.sorted_frequencies), matrix

    def nearest_frequency(self, target: float) -> Optional[float]:
        if not self.sorted_frequencies:
            return None
        arr = np.array(self.sorted_frequencies)
        return float(arr[np.argmin(np.abs(arr - target))])

    def field_slice_at_frequency(self, target_freq: float):
        """Return (H_field, processed_signal, actual_frequency) for the
        frequency nearest to ``target_freq``.
        """
        freq = self.nearest_frequency(target_freq)
        if freq is None:
            return None, None, None
        return self.H_field_by_freq[freq], self.processed[freq], freq

    def frequency_slice_at_field(self, target_field: float):
        """Return (frequencies, signal_at_field, actual_field) by taking,
        for every frequency, the processed value nearest to
        ``target_field`` on that frequency's own field axis.
        """
        if not self.sorted_frequencies:
            return None, None, None

        freqs = np.array(self.sorted_frequencies)
        values = np.zeros_like(freqs, dtype=float)
        actual_fields = np.zeros_like(freqs, dtype=float)

        for i, freq in enumerate(self.sorted_frequencies):
            H = self.H_field_by_freq[freq]
            idx = int(np.argmin(np.abs(H - target_field)))
            values[i] = self.processed[freq][idx]
            actual_fields[i] = H[idx]

        actual_field = float(np.median(actual_fields))
        return freqs, values, actual_field


def process_dataset(dataset: Dataset, settings: PreprocessSettings) -> ProcessedDataset:
    """Run :func:`preprocessing.preprocess_trace` over every frequency in a
    loaded :class:`loader.Dataset`, returning a :class:`ProcessedDataset`.
    """
    result = ProcessedDataset(label=dataset.label)
    result.sorted_frequencies = list(dataset.sorted_frequencies)

    for freq in result.sorted_frequencies:
        record = dataset.records[freq]

        # Use whichever channel carries the larger swing as the primary
        # signal. This mirrors the X/Y "transform" idea from the original
        # script without baking in a specific lock-in convention.
        if np.ptp(record.voltage_y) > np.ptp(record.voltage_x):
            primary_signal = record.voltage_y
        else:
            primary_signal = record.voltage_x

        bg_H = None
        bg_signal = None
        if settings.use_background and record.bg_voltage_x is not None:
            bg_H = record.H_field  # background interpolated onto same axis downstream
            bg_signal = (record.bg_voltage_y
                         if np.ptp(record.bg_voltage_y) > np.ptp(record.bg_voltage_x)
                         else record.bg_voltage_x)

        try:
            processed = preprocess_trace(
                record.H_field, primary_signal, settings,
                bg_H_field=bg_H, bg_signal=bg_signal,
            )
        except Exception as exc:
            result.warnings.append(f"Failed to process {freq:g} GHz ({record.filename}): {exc}")
            continue

        result.H_field_by_freq[freq] = record.H_field
        result.processed[freq] = processed

    result.sorted_frequencies = sorted(result.processed.keys())
    return result
