from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from auto_capcut.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Auto CapCut")
    app.setOrganizationName("AutoCapCut")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

