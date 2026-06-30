"""
gui/import_tab.py

Tab 1: Data Import.

- Experimental Datasets table: add/remove/reorder directories, assign a
  label to each (used later in plot legends and exports).
- Background Dataset section: pick one background directory, toggle
  background subtraction on/off.
- Data Summary: per-dataset file count / frequency range / field range.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QLineEdit, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from processing.loader import attach_background, load_dataset


class ImportTab(QWidget):

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._build_ui()
        self.app_state.datasets_changed.connect(self._refresh_summary)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QVBoxLayout(self)

        root.addWidget(self._build_experiment_group())
        root.addWidget(self._build_background_group())
        root.addWidget(self._build_summary_group())

    def _build_experiment_group(self) -> QGroupBox:
        group = QGroupBox("Experimental Datasets")
        layout = QVBoxLayout(group)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Directory", "Label"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Directory...")
        remove_btn = QPushButton("Remove Selected")
        up_btn = QPushButton("Move Up")
        down_btn = QPushButton("Move Down")
        load_btn = QPushButton("Load / Reload All")

        add_btn.clicked.connect(self._add_directory)
        remove_btn.clicked.connect(self._remove_selected)
        up_btn.clicked.connect(lambda: self._move_selected(-1))
        down_btn.clicked.connect(lambda: self._move_selected(1))
        load_btn.clicked.connect(self._load_all)

        for b in (add_btn, remove_btn, up_btn, down_btn, load_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        return group

    def _build_background_group(self) -> QGroupBox:
        group = QGroupBox("Background Dataset")
        layout = QHBoxLayout(group)

        self.bg_path_edit = QLineEdit()
        self.bg_path_edit.setReadOnly(True)
        self.bg_path_edit.setPlaceholderText("No background directory selected")

        browse_btn = QPushButton("Select Background Directory...")
        browse_btn.clicked.connect(self._select_background)

        self.bg_checkbox = QCheckBox("Use Background Subtraction")
        self.bg_checkbox.toggled.connect(self._toggle_background)

        layout.addWidget(self.bg_path_edit, stretch=1)
        layout.addWidget(browse_btn)
        layout.addWidget(self.bg_checkbox)

        return group

    def _build_summary_group(self) -> QGroupBox:
        group = QGroupBox("Data Summary")
        layout = QVBoxLayout(group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        return group

    # ------------------------------------------------------------- actions

    def _add_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Experimental Directory")
        if not directory:
            return
        label = directory.rstrip("/\\").split("/")[-1].split("\\")[-1]
        self.app_state.add_experiment(directory, label)
        self._reload_table()

    def _remove_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.app_state.remove_experiment(row)
        self._reload_table()

    def _move_selected(self, delta: int):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not rows:
            return
        row = rows[0]
        new_row = row + delta
        self.app_state.move_experiment(row, new_row)
        self._reload_table()
        self.table.selectRow(max(0, min(new_row, self.table.rowCount() - 1)))

    def _on_item_changed(self, item: QTableWidgetItem):
        if item.column() == 1:  # label column
            row = item.row()
            if 0 <= row < len(self.app_state.experiments):
                self.app_state.experiments[row].label = item.text()

    def _select_background(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Background Directory")
        if not directory:
            return
        self.app_state.background_directory = directory
        self.bg_path_edit.setText(directory)

    def _toggle_background(self, checked: bool):
        self.app_state.use_background_subtraction = checked
        self.app_state.settings.use_background = checked

    def _reload_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.app_state.experiments))
        for row, entry in enumerate(self.app_state.experiments):
            self.table.setItem(row, 0, QTableWidgetItem(entry.directory))
            self.table.setItem(row, 1, QTableWidgetItem(entry.label))
        self.table.blockSignals(False)

    def _load_all(self):
        if not self.app_state.experiments:
            QMessageBox.information(self, "Nothing to load", "Add at least one experimental directory first.")
            return

        if self.app_state.use_background_subtraction:
            if not self.app_state.background_directory:
                QMessageBox.warning(self, "Missing background",
                                     "Background subtraction is enabled but no background "
                                     "directory has been selected.")
            else:
                self.app_state.background_dataset = load_dataset(
                    self.app_state.background_directory, label="Background")

        warnings = []
        for entry in self.app_state.experiments:
            entry.dataset = load_dataset(entry.directory, label=entry.label)
            if self.app_state.background_dataset is not None:
                attach_background(entry.dataset, self.app_state.background_dataset)
            warnings.extend(f"[{entry.label}] {w}" for w in entry.dataset.warnings)

        self.app_state.datasets_changed.emit()
        self._refresh_summary()

        if warnings:
            QMessageBox.warning(self, "Load completed with warnings", "\n".join(warnings[:30]))

    def _refresh_summary(self):
        lines = []
        for entry in self.app_state.experiments:
            lines.append(f"=== {entry.label} ({entry.directory}) ===")
            if entry.dataset is None:
                lines.append("Not loaded yet.")
            else:
                lines.append(entry.dataset.summary_text())
            lines.append("")

        if self.app_state.background_dataset is not None:
            lines.append("=== Background ===")
            lines.append(self.app_state.background_dataset.summary_text())

        self.summary_text.setPlainText("\n".join(lines) if lines else "No datasets loaded yet.")
