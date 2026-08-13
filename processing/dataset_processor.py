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

# Upper bound on heatmap columns, so one unusually dense file cannot make the
# interpolated matrix enormous.
_MAX_GRID_POINTS = 20000


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
        """Build a common field axis used for the 2D heatmap matrix.

        The grid spans the intersection of the per-frequency field ranges and
        is sampled finely enough not to throw away resolution.

        Two things here used to be able to silently destroy the heatmap, and
        both are guarded now:

        * **Resolution.** The point count comes from the *widest* trace, not
          the median. A median is only meaningful when every record is a
          comparable sweep; if a run was aborted and left short stub files
          behind, and those stubs outnumber the real sweeps, the median
          collapses to the stub length and the whole map degenerates into a
          handful of columns stretched across the field axis.
        * **Degenerate records.** Records too short to interpolate are left
          out of the range and length statistics entirely, so a stub parked
          at the starting field cannot shrink - or invert - the common range
          for everyone else.
        """
        if not self.sorted_frequencies:
            return np.array([])

        mins, maxs, lengths = [], [], []
        for freq in self.sorted_frequencies:
            H = np.asarray(self.H_field_by_freq[freq], dtype=float)
            if len(H) < 2 or not np.isfinite(H).all():
                continue  # no usable range, and nothing to interpolate from
            mins.append(np.min(H))
            maxs.append(np.max(H))
            lengths.append(len(H))

        if not lengths:
            return np.array([])

        if n_points is None:
            # Preserve the finest resolution present, bounded so that one
            # pathological file cannot blow up memory.
            n_points = int(np.clip(max(lengths), 2, _MAX_GRID_POINTS))

        common_min = max(mins)
        common_max = min(maxs)
        if common_min >= common_max:
            # The traces share no common overlap. Span their union rather than
            # arbitrarily adopting the first trace's range, and let as_matrix()
            # mark the regions each trace never measured as missing.
            self.warnings.append(
                "Field ranges do not overlap across all frequencies (largest "
                f"minimum {common_min:.4g} >= smallest maximum {common_max:.4g}); "
                "heatmap spans their union and leaves unmeasured regions blank."
            )
            return np.linspace(min(mins), max(maxs), n_points)
        return np.linspace(common_min, common_max, n_points)

    def as_matrix(self, field_grid: Optional[np.ndarray] = None):
        """Interpolate every frequency's trace onto a common field grid and
        stack into a 2D matrix of shape (n_frequencies, n_field_points).

        Grid points outside a trace's own measured field range become NaN
        rather than being clamped to its endpoint value. Clamping invents flat
        bands of colour that read as real - and very strong - signal; NaN
        renders as blank, so a coverage gap looks like a gap.

        Returns (field_grid, freq_array, matrix).
        """
        if field_grid is None:
            field_grid = self.common_field_grid()

        matrix = np.full((len(self.sorted_frequencies), len(field_grid)), np.nan)
        for i, freq in enumerate(self.sorted_frequencies):
            H = np.asarray(self.H_field_by_freq[freq], dtype=float)
            sig = np.asarray(self.processed[freq], dtype=float)
            if len(H) < 2:
                continue
            # np.interp needs an ascending x-axis. A reverse sweep (+max down
            # to -max) arrives descending, so sort field and signal together.
            order = np.argsort(H)
            matrix[i, :] = np.interp(field_grid, H[order], sig[order],
                                     left=np.nan, right=np.nan)

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


def select_primary_signal(record) -> np.ndarray:
    """Pick whichever lock-in channel carries the larger swing as the
    primary signal. This mirrors the X/Y "transform" idea from the original
    script without baking in a specific lock-in convention.
    """
    if np.ptp(record.voltage_y) > np.ptp(record.voltage_x):
        return record.voltage_y
    return record.voltage_x


def process_record(record, settings: PreprocessSettings):
    """Run the full per-frequency pipeline on a single record.

    Steps: channel selection -> optional raw-data lineshape fit (the fitted
    curve replaces the raw trace; unfittable traces become zeros) ->
    :func:`preprocessing.preprocess_trace` (background subtraction, DC,
    detrend, Savitzky-Golay, quantity, enhancement).

    Returns ``(processed_array, warning_or_None)``.
    """
    primary_signal = select_primary_signal(record)
    warning = None

    if settings.use_lineshape_fit:
        from .lineshape_fitting import fit_best_lineshape
        fit = fit_best_lineshape(record.H_field, primary_signal,
                                 settings.lineshape_num_peaks)
        if fit.success:
            primary_signal = fit.fitted
        else:
            primary_signal = np.zeros_like(np.asarray(record.H_field, dtype=float))
            warning = f"Lineshape fit failed ({fit.message}); trace replaced with zeros."

    bg_H = None
    bg_signal = None
    if settings.use_background and record.bg_voltage_x is not None:
        bg_H = record.H_field  # background interpolated onto same axis downstream
        bg_signal = (record.bg_voltage_y
                     if np.ptp(record.bg_voltage_y) > np.ptp(record.bg_voltage_x)
                     else record.bg_voltage_x)

    processed = preprocess_trace(
        record.H_field, primary_signal, settings,
        bg_H_field=bg_H, bg_signal=bg_signal,
    )
    return processed, warning


def process_dataset(dataset: Dataset, settings: PreprocessSettings) -> ProcessedDataset:
    """Run the per-frequency pipeline (see :func:`process_record`) over
    every frequency in a loaded :class:`loader.Dataset`.
    """
    result = ProcessedDataset(label=dataset.label)
    result.sorted_frequencies = list(dataset.sorted_frequencies)

    for freq in result.sorted_frequencies:
        record = dataset.records[freq]

        try:
            processed, warning = process_record(record, settings)
        except Exception as exc:
            result.warnings.append(f"Failed to process {freq:g} GHz ({record.filename}): {exc}")
            continue

        if warning:
            result.warnings.append(f"{freq:g} GHz: {warning}")

        result.H_field_by_freq[freq] = record.H_field
        result.processed[freq] = processed

    result.sorted_frequencies = sorted(result.processed.keys())
    return result
