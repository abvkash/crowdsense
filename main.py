#!/usr/bin/env python3

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from app import CrowdSenseApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))

    window = CrowdSenseApp()
    window.show()
    
    sys.exit(app.exec())