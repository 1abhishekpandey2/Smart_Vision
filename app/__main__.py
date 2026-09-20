import sys

# ==========================================================
# WINDOWS NATIVE LIBRARY LOAD ORDER
# ==========================================================
#
# In the current Smart Vision Windows environment:
#
#     PyQt5 -> torch
#
# causes:
#
#     WinError 1114
#     c10.dll initialization failure
#
# while:
#
#     torch -> PyQt5
#
# works correctly.
#
# Therefore PyTorch must be loaded before any PyQt5 module.
#
# Do not move this import below the PyQt imports.
#
import torch  # noqa: F401


from PyQt5.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.ui.styles import APP_STYLE


def main():
    app = QApplication(sys.argv)

    app.setApplicationName("Smart Vision")

    app.setStyleSheet(APP_STYLE)

    window = MainWindow()

    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
