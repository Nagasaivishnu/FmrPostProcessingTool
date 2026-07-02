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

    def __init__(self):
        super().__init__()

        self.experiments: List[ExperimentEntry] = []
        self.background_directory: Optional[str] = None
        self.background_dataset: Optional[Dataset] = None
        self.use_background_subtraction: bool = False

        self.settings = PreprocessSettings()

        # Global display ranges shared by every tab's plots. ``None`` means
        # "auto" (let Matplotlib pick). Each is a (min, max) tuple.
        self.freq_range: Optional[Tuple[float, float]] = None
        self.field_range: Optional[Tuple[float, float]] = None

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
