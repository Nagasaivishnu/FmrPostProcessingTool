"""
main.py

Entry point for the Broadband FMR Post-Processing GUI application.

Run with:
    python main.py

Requires: PyQt5, matplotlib, numpy, scipy, pandas
(install with: pip install PyQt5 matplotlib numpy scipy pandas)
"""

import sys

from PyQt5.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
