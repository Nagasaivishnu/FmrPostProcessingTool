"""
gui/main_window.py

Top-level QMainWindow: assembles all five tabs around one shared AppState.
Styling is intentionally minimal (native Qt widgets, default look) - no
custom stylesheet theming, consistent with the earlier post-processing app.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QMainWindow, QTabWidget

from gui.app_state import AppState
from gui.export_tab import ExportTab
from gui.heatmap_tab import HeatmapTab
from gui.import_tab import ImportTab
from gui.processing_tab import ProcessingTab
from gui.section_tab import SectionTab


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Broadband FMR Post-Processing")
        self.resize(1280, 860)

        self.app_state = AppState()

        tabs = QTabWidget()
        tabs.addTab(ImportTab(self.app_state), "1. Data Import")
        tabs.addTab(ProcessingTab(self.app_state), "2. Processing Settings")
        tabs.addTab(HeatmapTab(self.app_state), "3. Heatmap Visualization")
        tabs.addTab(SectionTab(self.app_state), "4. Heatmap Sections")
        tabs.addTab(ExportTab(self.app_state), "5. Data Export")

        self.setCentralWidget(tabs)
