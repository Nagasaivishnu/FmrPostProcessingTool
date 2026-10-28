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

import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QLineEdit, QListView, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QTextEdit, QTreeView, QVBoxLayout, QWidget,
)


def _has_data_files(folder: Path) -> bool:
    """True if ``folder`` (or anything below it) holds files the loader
    would read: .csv files, or .txt files with 'FMR' in the name.
    """
    for _ in folder.glob("**/*.csv"):
        return True
    for _ in folder.glob("**/*FMR*.txt"):
        return True
    return False

from processing.loader import attach_background, load_dataset
from utils.dataset_list import load_dataset_list, save_dataset_list


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
        add_multi_btn = QPushButton("Add Multiple Directories...")
        add_multi_btn.setToolTip("Pick several folders at once (Ctrl / Shift + click)")
        add_sub_btn = QPushButton("Add All Subfolders...")
        add_sub_btn.setToolTip("Pick a parent folder; every subfolder containing "
                               "data files is added as its own dataset")
        remove_btn = QPushButton("Remove Selected")
        up_btn = QPushButton("Move Up")
        down_btn = QPushButton("Move Down")
        load_btn = QPushButton("Load / Reload All")

        add_btn.clicked.connect(self._add_directory)
        add_multi_btn.clicked.connect(self._add_multiple_directories)
        add_sub_btn.clicked.connect(self._add_subfolders)
        remove_btn.clicked.connect(self._remove_selected)
        up_btn.clicked.connect(lambda: self._move_selected(-1))
        down_btn.clicked.connect(lambda: self._move_selected(1))
        load_btn.clicked.connect(self._load_all)

        add_row = QHBoxLayout()
        for b in (add_btn, add_multi_btn, add_sub_btn):
            add_row.addWidget(b)
        layout.addLayout(add_row)

        for b in (remove_btn, up_btn, down_btn, load_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        # Save / reopen the whole list (directories + labels + background).
        list_row = QHBoxLayout()
        open_list_btn = QPushButton("Open Dataset List (.yaml)...")
        open_list_btn.setToolTip("Load every directory and label saved in a list file")
        save_list_btn = QPushButton("Save Dataset List (.yaml)...")
        save_list_btn.setToolTip("Save all directories, labels and the background "
                                 "directory to a file you can reopen later")
        self.autoload_checkbox = QCheckBox("Load data after opening list")
        self.autoload_checkbox.setChecked(True)
        open_list_btn.clicked.connect(self._open_dataset_list)
        save_list_btn.clicked.connect(self._save_dataset_list)
        list_row.addWidget(open_list_btn)
        list_row.addWidget(save_list_btn)
        list_row.addWidget(self.autoload_checkbox)
        list_row.addStretch(1)
        layout.addLayout(list_row)

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
        self._add_directories([directory])

    def _add_multiple_directories(self):
        """Folder picker that allows selecting several folders at once.
        The native Windows folder dialog only allows one, so this uses Qt's
        own dialog with multi-selection switched on.
        """
        dialog = QFileDialog(self, "Select Experimental Directories (Ctrl/Shift + click)")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        for view in dialog.findChildren(QListView) + dialog.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if not dialog.exec_():
            return
        # The dialog may also report the folder it was browsing; drop it if
        # its own subfolders were picked.
        picked = [os.path.normpath(p) for p in dialog.selectedFiles()]
        current = os.path.normpath(dialog.directory().absolutePath())
        if len(picked) > 1 and current in picked:
            picked.remove(current)
        self._add_directories(picked)

    def _add_subfolders(self):
        parent = QFileDialog.getExistingDirectory(
            self, "Select Parent Folder (each data subfolder becomes a dataset)")
        if not parent:
            return
        subfolders = sorted((p for p in Path(parent).iterdir()
                             if p.is_dir() and _has_data_files(p)),
                            key=lambda p: p.name.lower())
        if not subfolders:
            QMessageBox.information(
                self, "No data folders",
                f"No subfolders with .csv / FMR .txt files were found in:\n{parent}")
            return
        self._add_directories([str(p) for p in subfolders])

    def _add_directories(self, directories):
        """Add folders as datasets: skip ones already in the table, label
        each by its folder name, and keep labels unique (labels are used
        as keys for processed results).
        """
        existing_dirs = {os.path.normcase(os.path.normpath(e.directory))
                         for e in self.app_state.experiments}
        existing_labels = {e.label for e in self.app_state.experiments}

        added, skipped = [], []
        for directory in directories:
            key = os.path.normcase(os.path.normpath(directory))
            if key in existing_dirs:
                skipped.append(directory)
                continue
            base = os.path.basename(os.path.normpath(directory)) or directory
            label, n = base, 2
            while label in existing_labels:
                label = f"{base} ({n})"
                n += 1
            self.app_state.add_experiment(directory, label)
            existing_dirs.add(key)
            existing_labels.add(label)
            added.append(label)

        self._reload_table()

        if len(directories) > 1 or skipped:
            msg = f"Added {len(added)} dataset(s)."
            if skipped:
                msg += (f"\n\nSkipped {len(skipped)} already in the list:\n"
                        + "\n".join(skipped[:15]))
            QMessageBox.information(self, "Directories added", msg)

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

    # ---------------------------------------------------- dataset list file

    def _save_dataset_list(self):
        if not self.app_state.experiments:
            QMessageBox.information(self, "Nothing to save",
                                     "Add at least one experimental directory first.")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save Dataset List", "datasets.yaml", "YAML (*.yaml *.yml)")
        if not path:
            return
        if not path.lower().endswith((".yaml", ".yml")):
            path += ".yaml"
        try:
            save_dataset_list(
                path,
                [(e.label, e.directory) for e in self.app_state.experiments],
                background_dir=self.app_state.background_directory,
                use_background=self.app_state.use_background_subtraction,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(
            self, "Dataset list saved",
            f"Saved {len(self.app_state.experiments)} dataset(s) to:\n{path}")

    def _open_dataset_list(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Open Dataset List", "", "YAML (*.yaml *.yml);;All files (*)")
        if not path:
            return
        try:
            datasets, bg_dir, use_bg = load_dataset_list(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not read list", str(exc))
            return
        if not datasets:
            QMessageBox.information(self, "Empty list", "The file contains no datasets.")
            return

        # Replace or add to what's already in the table?
        if self.app_state.experiments:
            box = QMessageBox(self)
            box.setWindowTitle("Open Dataset List")
            box.setText(f"The table already has {len(self.app_state.experiments)} "
                        f"dataset(s).\nReplace them with the {len(datasets)} from the "
                        "file, or add the file's datasets to the end?")
            replace_btn = box.addButton("Replace", QMessageBox.AcceptRole)
            add_btn = box.addButton("Add to List", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() == replace_btn:
                self.app_state.experiments.clear()
                self.app_state.datasets_changed.emit()
            elif box.clickedButton() != add_btn:
                return

        missing = [d for _label, d in datasets if not os.path.isdir(d)]
        existing_dirs = {os.path.normcase(os.path.normpath(e.directory))
                         for e in self.app_state.experiments}
        existing_labels = {e.label for e in self.app_state.experiments}
        added = skipped = 0
        for label, directory in datasets:
            if directory in missing:
                continue
            key = os.path.normcase(os.path.normpath(directory))
            if key in existing_dirs:
                skipped += 1
                continue
            base, n = label, 2
            while label in existing_labels:  # labels must stay unique
                label = f"{base} ({n})"
                n += 1
            self.app_state.add_experiment(directory, label)
            existing_dirs.add(key)
            existing_labels.add(label)
            added += 1

        if bg_dir:
            if os.path.isdir(bg_dir):
                self.app_state.background_directory = bg_dir
                self.bg_path_edit.setText(bg_dir)
                self.bg_checkbox.setChecked(use_bg)
            else:
                missing.append(f"{bg_dir}  (background)")

        self._reload_table()

        msg = f"Added {added} dataset(s) from:\n{path}"
        if skipped:
            msg += f"\n\nSkipped {skipped} already in the list."
        if missing:
            msg += ("\n\nThese folders were not found and were skipped:\n"
                    + "\n".join(missing[:15]))
            QMessageBox.warning(self, "Dataset list opened", msg)
        elif not (self.autoload_checkbox.isChecked() and added):
            QMessageBox.information(self, "Dataset list opened", msg)

        if self.autoload_checkbox.isChecked() and added:
            self._load_all()

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
