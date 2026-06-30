"""
plotting/mpl_canvas.py

A small reusable Matplotlib-in-Qt widget: a FigureCanvas + NavigationToolbar
(gives zoom/pan/save/reset for free) plus a crosshair coordinate readout,
used by the Heatmap and Section tabs.
"""

from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class MplCanvas(QWidget):
    """A figure + toolbar + coordinate readout, packaged as one QWidget."""

    def __init__(self, parent=None, figsize=(7, 5)):
        super().__init__(parent)

        self.figure = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.coord_label = QLabel("x: -, y: -")
        self.coord_label.setStyleSheet("color: #666;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.coord_label)

        self.canvas.mpl_connect("motion_notify_event", self._on_move)

    def _on_move(self, event):
        if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
            self.coord_label.setText(f"x: {event.xdata:.6g}, y: {event.ydata:.6g}")

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def draw(self):
        self.canvas.draw_idle()
