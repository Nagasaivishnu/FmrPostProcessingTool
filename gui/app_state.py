"""
gui/app_state.py

A single shared state object passed between tabs. Holds the list of loaded
experimental datasets (with user labels), the optional background dataset,
the current preprocessing settings, and the resulting processed datasets.

Using one shared state object (instead of tabs reaching into each other
directly) keeps the tabs loosely coupled: each tab reads/writes app_state
and the state emits Qt signals so other tabs know to refresh.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from processing.dataset_processor import ProcessedDataset
from processing.loader import Dataset
from processing.preprocessing import PreprocessSettings
from plotting.heatmap_render import HeatmapStyle

DEFAULT_HEATMAP_CMAP = "viridis"

# Magnetic-field display units. Raw files are read in tesla; the value is
# how many of that unit make 1 T (1 T = 1000 mT = 10 000 Oe).
FIELD_UNITS = {"T": 1.0, "mT": 1.0e3, "Oe": 1.0e4}


class ExperimentEntry:
    """One row in the experimental-datasets table: a loaded Dataset plus
    its user-assigned label and directory path.
    """

    def __init__(self, directory: str, label: str, dataset: Optional[Dataset] = None):
        self.directory = directory
        self.label = label
        self.dataset: Optional[Dataset] = dataset

    def __repr__(self):
        return f"ExperimentEntry(label={self.label!r}, directory={self.directory!r})"


class AppState(QObject):
    """Central, Qt-signal-emitting store of everything the GUI needs to
    share across tabs.
    """

    # Emitted whenever the list of experiments (or their loaded data) changes.
    datasets_changed = pyqtSignal()
    # Emitted whenever processed results are (re)computed.
    processed_changed = pyqtSignal()
    # Emitted whenever the global display ranges (frequency/field) change.
    display_range_changed = pyqtSignal()
    # Emitted when the field unit changes; argument = new/old scale ratio
    # (e.g. 10000.0 for T -> Oe) so tabs can rescale field-valued inputs.
    field_unit_changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()

        self.experiments: List[ExperimentEntry] = []
        self.background_directory: Optional[str] = None
        self.background_dataset: Optional[Dataset] = None
        self.use_background_subtraction: bool = False

        # Unit used for magnetic field everywhere in the GUI (plots, inputs,
        # exports). Processed data are stored already converted to it.
        self.field_unit: str = "T"

        self.settings = PreprocessSettings()

        # Global display ranges shared by every tab's plots. ``None`` means
        # "auto" (let Matplotlib pick). Each is a (min, max) tuple.
        self.freq_range: Optional[Tuple[float, float]] = None
        self.field_range: Optional[Tuple[float, float]] = None

        # label -> ProcessedDataset
        self.processed: Dict[str, ProcessedDataset] = {}

        # label -> Matplotlib colormap name used for that dataset's heatmap.
        # Each folder/dataset remembers its own choice; datasets without an
        # entry use DEFAULT_HEATMAP_CMAP.
        self.heatmap_cmaps: Dict[str, str] = {}

        # Display settings shared by every heatmap figure (screen, PNG
        # batch save, Export tab): scaling, fonts, Origin style, aspect.
        self.heatmap_style = HeatmapStyle()

    # --- Experiment list helpers -------------------------------------------------

    def add_experiment(self, directory: str, label: str) -> None:
        self.experiments.append(ExperimentEntry(directory=directory, label=label))
        self.datasets_changed.emit()

    def remove_experiment(self, index: int) -> None:
        if 0 <= index < len(self.experiments):
            del self.experiments[index]
            self.datasets_changed.emit()

    def move_experiment(self, index: int, new_index: int) -> None:
        if 0 <= index < len(self.experiments) and 0 <= new_index < len(self.experiments):
            entry = self.experiments.pop(index)
            self.experiments.insert(new_index, entry)
            self.datasets_changed.emit()

    def labels(self) -> List[str]:
        return [e.label for e in self.experiments]

    def get_processed(self, label: str) -> Optional[ProcessedDataset]:
        return self.processed.get(label)

    def processed_labels(self) -> List[str]:
        return list(self.processed.keys())

    # --- Per-dataset heatmap colormaps ------------------------------------------

    def get_heatmap_cmap(self, label: str) -> str:
        return self.heatmap_cmaps.get(label, DEFAULT_HEATMAP_CMAP)

    def set_heatmap_cmap(self, label: str, cmap: str) -> None:
        if label:
            self.heatmap_cmaps[label] = cmap

    # --- Field unit ---------------------------------------------------------------

    @property
    def field_factor(self) -> float:
        """Multiply a field in tesla by this to get the current unit."""
        return FIELD_UNITS[self.field_unit]

    def field_axis_label(self, base: str = "Magnetic Field") -> str:
        return f"{base} ({self.field_unit})"

    def apply_field_unit(self, result: ProcessedDataset) -> ProcessedDataset:
        """Convert a freshly processed result (fields in tesla) to the
        current unit. New arrays are made so the raw loaded data stay in T.
        """
        f = self.field_factor
        if f != 1.0:
            result.H_field_by_freq = {k: np.asarray(v, dtype=float) * f
                                      for k, v in result.H_field_by_freq.items()}
        return result

    def set_field_unit(self, unit: str) -> None:
        """Switch the field unit: rescale every processed dataset and the
        field display range, then tell all tabs.
        """
        if unit not in FIELD_UNITS or unit == self.field_unit:
            return
        ratio = FIELD_UNITS[unit] / FIELD_UNITS[self.field_unit]
        self.field_unit = unit
        for result in self.processed.values():
            result.H_field_by_freq = {k: np.asarray(v, dtype=float) * ratio
                                      for k, v in result.H_field_by_freq.items()}
        if self.field_range is not None:
            self.field_range = (self.field_range[0] * ratio, self.field_range[1] * ratio)
        self.field_unit_changed.emit(ratio)
        self.processed_changed.emit()
        self.display_range_changed.emit()

    # --- Global display ranges ---------------------------------------------------

    def set_display_ranges(self, freq_range: Optional[Tuple[float, float]],
                           field_range: Optional[Tuple[float, float]]) -> None:
        """Set the shared frequency/field display ranges (each a (min, max)
        tuple or ``None`` for auto) and notify all tabs.
        """
        self.freq_range = freq_range
        self.field_range = field_range
        self.display_range_changed.emit()

    def apply_ranges_to_axes(self, ax, x_kind: Optional[str] = None,
                             y_kind: Optional[str] = None) -> None:
        """Apply the shared ranges to a Matplotlib Axes. ``x_kind``/``y_kind``
        say what quantity each axis represents: ``"field"``, ``"frequency"``,
        or ``None`` (leave that axis alone). Ranges that are ``None`` (auto)
        are skipped.
        """
        ranges = {"field": self.field_range, "frequency": self.freq_range}
        xr = ranges.get(x_kind)
        if xr is not None:
            ax.set_xlim(xr[0], xr[1])
        yr = ranges.get(y_kind)
        if yr is not None:
            ax.set_ylim(yr[0], yr[1])


def rescale_field_spin(spin, ratio: float) -> None:
    """Convert a field-valued QDoubleSpinBox to a new unit (value, range and
    step all multiplied by ``ratio``) without firing valueChanged.
    """
    spin.blockSignals(True)
    lo, hi, val = spin.minimum(), spin.maximum(), spin.value()
    spin.setRange(lo * ratio, hi * ratio)
    spin.setSingleStep(spin.singleStep() * ratio)
    spin.setValue(val * ratio)
    spin.blockSignals(False)
