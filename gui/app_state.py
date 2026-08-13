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

from PyQt5.QtCore import QObject, pyqtSignal

from processing.dataset_processor import ProcessedDataset
from processing.loader import Dataset
from processing.preprocessing import PreprocessSettings


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
    # Emitted whenever the magnetic-field display unit changes (T <-> Oe).
    field_unit_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.experiments: List[ExperimentEntry] = []
        self.background_directory: Optional[str] = None
        self.background_dataset: Optional[Dataset] = None
        self.use_background_subtraction: bool = False

        self.settings = PreprocessSettings()

        # Global display ranges shared by every tab's plots. ``None`` means
        # "auto" (let Matplotlib pick). Each is a (min, max) tuple.
        # Field-range values are entered/stored in Tesla.
        self.freq_range: Optional[Tuple[float, float]] = None
        self.field_range: Optional[Tuple[float, float]] = None

        # Magnetic-field display unit. Raw data is in Tesla; when set to
        # "Oe", every plotted field value is multiplied by 1e4 for display
        # (data, inputs, and exports remain in Tesla).
        self.field_unit: str = "T"

        # label -> ProcessedDataset
        self.processed: Dict[str, ProcessedDataset] = {}

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

    # --- Field display unit -------------------------------------------------------

    def set_field_unit(self, unit: str) -> None:
        """Set the magnetic-field display unit ("T" or "Oe") and notify all
        tabs. Raw data stays in Tesla; conversion happens at plot time.
        """
        unit = "Oe" if str(unit).lower().startswith("oe") else "T"
        if unit != self.field_unit:
            self.field_unit = unit
            self.field_unit_changed.emit()

    def field_scale(self) -> float:
        """Multiplier applied to Tesla field values for display
        (1.0 for T, 1e4 for Oe).
        """
        return 1.0e4 if self.field_unit == "Oe" else 1.0

    def field_unit_label(self) -> str:
        return self.field_unit

    def field_axis_label(self) -> str:
        return f"Magnetic Field ({self.field_unit})"

    # --- Global display ranges ---------------------------------------------------

    def set_display_ranges(self, freq_range: Optional[Tuple[float, float]],
                           field_range: Optional[Tuple[float, float]]) -> None:
        """Set the shared frequency/field display ranges (each a (min, max)
        tuple or ``None`` for auto) and notify all tabs. Field values are
        in Tesla.
        """
        self.freq_range = freq_range
        self.field_range = field_range
        self.display_range_changed.emit()

    def apply_ranges_to_axes(self, ax, x_kind: Optional[str] = None,
                             y_kind: Optional[str] = None) -> None:
        """Apply the shared ranges to a Matplotlib Axes. ``x_kind``/``y_kind``
        say what quantity each axis represents: ``"field"``, ``"frequency"``,
        or ``None`` (leave that axis alone). Ranges that are ``None`` (auto)
        are skipped. Field limits (stored in Tesla) are converted to the
        current display unit, matching the plotted (scaled) data.
        """
        s = self.field_scale()
        field_r = (self.field_range[0] * s, self.field_range[1] * s) \
            if self.field_range is not None else None
        ranges = {"field": field_r, "frequency": self.freq_range}
        xr = ranges.get(x_kind)
        if xr is not None:
            ax.set_xlim(xr[0], xr[1])
        yr = ranges.get(y_kind)
        if yr is not None:
            ax.set_ylim(yr[0], yr[1])
