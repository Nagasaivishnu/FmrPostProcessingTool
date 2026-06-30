"""
processing/peak_analysis.py

Peak detection on processed FMR data: for each frequency's field-sweep
cross-section, find the strongest N peaks and report their field
positions, so the field-vs-frequency peak "branches" (resonance tracks)
can be plotted and exported.

Kept Qt-free, same as the rest of the processing/ package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from .dataset_processor import ProcessedDataset


def validate_num_peaks(num_peaks: int) -> Optional[str]:
    """Return an error string if ``num_peaks`` is invalid, else ``None``."""
    if num_peaks < 1:
        return "Number of peaks must be at least 1."
    return None


def find_peak_positions(H_field: np.ndarray, signal: np.ndarray,
                         num_peaks: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find the ``num_peaks`` strongest local maxima in a single
    field-sweep cross-section.

    The strongest peaks (by signal height) are selected first, then
    re-ordered by ascending field position so that, across a frequency
    sweep, "peak 1" / "peak 2" / ... consistently refer to the same
    low-field-to-high-field branch rather than jumping around by
    amplitude rank. If fewer than ``num_peaks`` local maxima exist, the
    remaining slots are filled with NaN.

    Returns
    -------
    (fields, heights) : both arrays of length ``num_peaks``
    """
    fields_out = np.full(num_peaks, np.nan)
    heights_out = np.full(num_peaks, np.nan)

    if len(signal) < 3:
        return fields_out, heights_out

    peak_idx, _properties = find_peaks(signal)
    if len(peak_idx) == 0:
        return fields_out, heights_out

    heights = signal[peak_idx]
    # Strongest first.
    strength_order = np.argsort(heights)[::-1][:num_peaks]
    selected_idx = peak_idx[strength_order]

    selected_fields = H_field[selected_idx]
    selected_heights = signal[selected_idx]

    # Re-order the selected peaks by field position so peak tracks stay
    # consistent across frequency.
    field_order = np.argsort(selected_fields)
    selected_fields = selected_fields[field_order]
    selected_heights = selected_heights[field_order]

    n_found = len(selected_fields)
    fields_out[:n_found] = selected_fields
    heights_out[:n_found] = selected_heights

    return fields_out, heights_out


@dataclass
class PeakResult:
    """Peak-tracking result for one processed dataset.

    ``peak_fields`` and ``peak_heights`` have shape
    ``(num_peaks, n_frequencies)``; row *i* is the field-position track
    (and corresponding signal height) of the *(i+1)*-th peak, ordered by
    ascending field position at each frequency (see
    :func:`find_peak_positions`).
    """

    label: str
    num_peaks: int
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_fields: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_heights: np.ndarray = field(default_factory=lambda: np.array([]))
    warnings: List[str] = field(default_factory=list)

    def track(self, peak_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, field_positions) for one peak track,
        dropping frequencies where that peak wasn't found (NaN).
        """
        fields = self.peak_fields[peak_index, :]
        mask = ~np.isnan(fields)
        return self.frequencies[mask], fields[mask]


def compute_peaks(result: ProcessedDataset, num_peaks: int) -> PeakResult:
    """Run peak detection over every frequency in a processed dataset.

    Parameters
    ----------
    result : ProcessedDataset
        Output of :func:`processing.dataset_processor.process_dataset`.
    num_peaks : int
        Number of peaks to track per frequency cross-section (user input).
    """
    peak_result = PeakResult(label=result.label, num_peaks=num_peaks)

    err = validate_num_peaks(num_peaks)
    if err:
        peak_result.warnings.append(err)
        return peak_result

    freqs = np.array(result.sorted_frequencies)
    if len(freqs) == 0:
        peak_result.warnings.append("No processed frequencies available - run Process All Datasets first.")
        return peak_result

    peak_fields = np.full((num_peaks, len(freqs)), np.nan)
    peak_heights = np.full((num_peaks, len(freqs)), np.nan)

    n_missing = 0
    for i, freq in enumerate(result.sorted_frequencies):
        H = result.H_field_by_freq[freq]
        sig = result.processed[freq]
        fields, heights = find_peak_positions(H, sig, num_peaks)
        peak_fields[:, i] = fields
        peak_heights[:, i] = heights
        if np.any(np.isnan(fields)):
            n_missing += 1

    if n_missing:
        peak_result.warnings.append(
            f"{n_missing} of {len(freqs)} frequencies had fewer than {num_peaks} "
            f"detectable peak(s); missing points are left blank in the plot/export."
        )

    peak_result.frequencies = freqs
    peak_result.peak_fields = peak_fields
    peak_result.peak_heights = peak_heights
    return peak_result
