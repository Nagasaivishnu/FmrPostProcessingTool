"""
processing/peak_analysis.py

Peak detection on processed FMR data: for each frequency's field-sweep
cross-section, find the strongest N peaks and report their field
positions, so the field-vs-frequency peak "branches" (resonance tracks)
can be plotted and exported.

Peaks 2..N can each have a maximum allowable field-gap from peak 1 (the
lowest-field peak). If a peak is found farther from peak 1 than its
allowed gap, it's treated as noise: its position is overlapped onto peak
1's position instead of being reported as a separate, far-flung point.

Kept Qt-free, same as the rest of the processing/ package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from .dataset_processor import ProcessedDataset

DEFAULT_MAX_GAP = 0.05  # Tesla


def validate_num_peaks(num_peaks: int) -> Optional[str]:
    """Return an error string if ``num_peaks`` is invalid, else ``None``."""
    if num_peaks < 1:
        return "Number of peaks must be at least 1."
    return None


def validate_max_gaps(num_peaks: int, max_gaps: Sequence[float]) -> Optional[str]:
    """Validate the list of per-peak max-gap-from-first-peak thresholds.

    There should be exactly ``num_peaks - 1`` thresholds (one for each
    peak after the first), and every threshold must be non-negative.
    """
    expected = max(num_peaks - 1, 0)
    if len(max_gaps) != expected:
        return f"Expected {expected} gap setting(s) for {num_peaks} peak(s), got {len(max_gaps)}."
    for i, gap in enumerate(max_gaps):
        if gap < 0:
            return f"Peak {i + 2} max gap must be non-negative."
    return None


def find_peak_positions(H_field: np.ndarray, signal: np.ndarray,
                         num_peaks: int,
                         max_gaps: Optional[Sequence[float]] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Find the ``num_peaks`` strongest local maxima in a single
    field-sweep cross-section.

    The strongest peaks (by signal height) are selected first, then
    re-ordered by ascending field position so that, across a frequency
    sweep, "peak 1" / "peak 2" / ... consistently refer to the same
    low-field-to-high-field branch rather than jumping around by
    amplitude rank. If fewer than ``num_peaks`` local maxima exist, the
    remaining slots are filled with NaN.

    Parameters
    ----------
    max_gaps : sequence of float, optional
        Maximum allowed field distance of peak *i* (i = 2..num_peaks)
        from peak 1, ``max_gaps[i - 2]``. If a found peak is farther from
        peak 1 than its allowed gap, it's treated as a noise peak: its
        field/height are overlapped onto peak 1's values instead of being
        reported as a separate point. Length must be ``num_peaks - 1``;
        pass ``None`` to skip gap filtering entirely.

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

    # Gap filtering: any peak farther from peak 1 than its allowed gap is
    # noise - overlap it onto peak 1 instead of reporting a stray point.
    if max_gaps is not None and n_found > 0 and not np.isnan(fields_out[0]):
        ref_field = fields_out[0]
        ref_height = heights_out[0]
        for i in range(1, num_peaks):
            if np.isnan(fields_out[i]):
                continue
            gap_limit = max_gaps[i - 1] if (i - 1) < len(max_gaps) else None
            if gap_limit is not None and abs(fields_out[i] - ref_field) > gap_limit:
                fields_out[i] = ref_field
                heights_out[i] = ref_height

    return fields_out, heights_out


@dataclass
class PeakResult:
    """Peak-tracking result for one processed dataset.

    ``peak_fields`` and ``peak_heights`` have shape
    ``(num_peaks, n_frequencies)``; row *i* is the field-position track
    (and corresponding signal height) of the *(i+1)*-th peak, ordered by
    ascending field position at each frequency (see
    :func:`find_peak_positions`). Peaks beyond their allowed gap from
    peak 1 have been overlapped onto peak 1's values rather than left as
    stray far-field points.
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


def compute_peaks(result: ProcessedDataset, num_peaks: int,
                   max_gaps: Optional[Sequence[float]] = None) -> PeakResult:
    """Run peak detection over every frequency in a processed dataset.

    Parameters
    ----------
    result : ProcessedDataset
        Output of :func:`processing.dataset_processor.process_dataset`.
    num_peaks : int
        Number of peaks to track per frequency cross-section (user input).
    max_gaps : sequence of float, optional
        Per-peak maximum allowed field distance from peak 1 (length
        ``num_peaks - 1``). Peaks farther than their limit are treated as
        noise and overlapped onto peak 1. ``None`` disables gap filtering.
    """
    peak_result = PeakResult(label=result.label, num_peaks=num_peaks)

    err = validate_num_peaks(num_peaks)
    if err:
        peak_result.warnings.append(err)
        return peak_result

    if max_gaps is not None:
        err = validate_max_gaps(num_peaks, max_gaps)
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
        fields, heights = find_peak_positions(H, sig, num_peaks, max_gaps=max_gaps)
        peak_fields[:, i] = fields
        peak_heights[:, i] = heights
        if np.any(np.isnan(fields)):
            n_missing += 1

    n_overlapped = 0
    if max_gaps is not None and num_peaks > 1:
        # Count how many (frequency, peak) points were pulled onto peak 1
        # by gap filtering, for a user-facing summary.
        ref = peak_fields[0:1, :]
        same_as_ref = np.isclose(peak_fields[1:, :], ref, equal_nan=False)
        n_overlapped = int(np.sum(same_as_ref & ~np.isnan(peak_fields[1:, :])))

    if n_missing:
        peak_result.warnings.append(
            f"{n_missing} of {len(freqs)} frequencies had fewer than {num_peaks} "
            f"detectable peak(s); missing points are left blank in the plot/export."
        )
    if n_overlapped:
        peak_result.warnings.append(
            f"{n_overlapped} peak point(s) exceeded their allowed gap from Peak 1 "
            f"and were treated as noise (overlapped onto Peak 1)."
        )

    peak_result.frequencies = freqs
    peak_result.peak_fields = peak_fields
    peak_result.peak_heights = peak_heights
    return peak_result
