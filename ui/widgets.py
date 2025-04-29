import sys
import os
import cv2
import torch
import numpy as np
import urllib.request

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QFrame, QSizePolicy, QFileDialog, QProgressBar, QSlider, QCheckBox, QDialog, QSplitter, QScrollArea, QGridLayout)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QSize, QThread, pyqtSignal, pyqtProperty, QRect, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent, QPainter, QPainterPath, QColor, QPen, QFontMetrics
from PyQt6.QtSvg import QSvgRenderer

from config import *

class ToggleSwitch(QWidget):
    """Modern toggle switch widget"""

    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(50, 24)
        self.setMaximumHeight(24)

        self.checked = False
        self.thumb_position = 2  # Initial position

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        """Custom paint event to draw the toggle switch"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        if not self.isEnabled():
            # Disabled state
            track_color = QColor('#3E3E42')  # Gray for disabled
            thumb_color = QColor('#888888')  # Light gray for disabled thumb
        else:
            # Enabled state
            track_color = QColor(ACCENT_COLOR) if self.checked else QColor(BORDER_COLOR)
            thumb_color = QColor('#FFFFFF') if self.checked else QColor('#AAAAAA')

        # Draw track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        track_rect = QRect(0, 0, width, height)
        painter.drawRoundedRect(track_rect, height//2, height//2)

        thumb_width = height - 4
        if self.checked:
            thumb_pos = width - thumb_width - 2
        else:
            thumb_pos = 2

        painter.setBrush(thumb_color)
        thumb_rect = QRect(thumb_pos, 2, thumb_width, thumb_width)
        painter.drawEllipse(thumb_rect)

    def mousePressEvent(self, event):
        """Handle mouse press events to toggle the switch"""
        if not self.isEnabled():
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.checked = not self.checked
            self.update()  # Force redraw
            self.toggled.emit(self.checked)
            event.accept()

    def setChecked(self, checked):
        """Set the checked state programmatically"""
        if self.checked != checked:
            self.checked = checked
            self.update()  # Force redraw

    def isChecked(self):
        """Return the current checked state"""
        return self.checked

class ModernBoxedSlider(QWidget):
    """Custom slider widget that looks like a filled progress bar with text inside"""

    valueChanged = pyqtSignal(int)

    def __init__(self, integer_display=False, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 32)
        self.setMaximumHeight(32)

        # Default slider properties
        self.minimum = 10
        self.maximum = 90
        self.value = 40
        self.pressed = False
        self.hover = False
        self.integer_display = integer_display
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def setValue(self, value):
        """Set slider value and emit change signal if needed"""
        value = max(self.minimum, min(self.maximum, value))
        if self.value != value:
            self.value = value
            self.update()  # Trigger repaint
            self.valueChanged.emit(value)

    def getValue(self):
        """Get current slider value"""
        return self.value

    def setRange(self, minimum, maximum):
        """Set slider range"""
        self.minimum = minimum
        self.maximum = maximum
        self.value = max(self.minimum, min(self.maximum, self.value))
        self.update()

    def paintEvent(self, event):
        """Custom paint event to draw the slider"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Calculate filled width
        value_range = self.maximum - self.minimum
        value_position = (self.value - self.minimum) / value_range if value_range > 0 else 0
        filled_width = int(width * value_position)

        # Draw background (unfilled part)
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        painter.setBrush(QColor(DROPDOWN_BG_COLOR))
        painter.drawRoundedRect(0, 0, width, height, 4, 4)

        # Draw filled part
        if filled_width > 0:
            # Use lighter blue when hovered
            fill_color = QColor(ACCENT_COLOR) if self.hover else QColor(DARKER_ACCENT_COLOR)
            painter.setBrush(fill_color)

            painter.setPen(QPen(QColor(LIGHTER_ACCENT_COLOR), 0.5))

            # Create a rectangular path for the filled portion
            filled_rect = QRect(0, 0, filled_width, height)

            # Handle the rounded corners
            if filled_width < width:
                # If not fully filled, use a clipped path to draw
                path = QPainterPath()
                path.addRoundedRect(0, 0, width, height, 4, 4)
                painter.setClipPath(path)
                painter.drawRect(filled_rect)
                painter.setClipping(False)
            else:
                # If fully filled, draw with rounded corners
                painter.drawRoundedRect(0, 0, filled_width, height, 4, 4)

        # Draw border again to ensure it's visible
        border_color = ACCENT_COLOR if self.hover else BORDER_COLOR
        painter.setPen(QPen(QColor(border_color), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(0, 0, width, height, 5, 5)

        if self.integer_display:
            value_text = str(self.value)
        else:
            value_text = f"{self.value / 100:.2f}"

        painter.setPen(QColor(TEXT_COLOR))
        font = QFont('Segoe UI', 14)
        painter.setFont(font)
        text_rect = QRect(0, 0, width, height)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, value_text)

    def mousePressEvent(self, event):
        """Handle mouse press event"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = True
            try:
                # PyQt6 style
                x = event.position().x()
            except:
                # PyQt5 compatibility
                x = event.x()
            self.updateValueFromMouse(x)
            self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move event"""
        try:
            # PyQt6 style
            x = event.position().x()
        except:
            # PyQt5 compatibility
            x = event.x()

        self.hover = True
        self.update()

        if self.pressed:
            self.updateValueFromMouse(x)

    def mouseReleaseEvent(self, event):
        """Handle mouse release event"""
        if event.button() == Qt.MouseButton.LeftButton and self.pressed:
            self.pressed = False
            try:
                # PyQt6 style
                x = event.position().x()
            except:
                # PyQt5 compatibility
                x = event.x()
            self.updateValueFromMouse(x)
            self.update()

    def leaveEvent(self, event):
        """Handle mouse leave event"""
        self.hover = False
        self.update()

    def updateValueFromMouse(self, x):
        """Update slider value based on mouse position"""
        value_range = self.maximum - self.minimum
        value_position = max(0, min(1, x / self.width()))
        new_value = self.minimum + int(value_position * value_range)
        self.setValue(new_value)

    def sizeHint(self):
        """Provide a default size hint"""
        return QSize(150, 32)

class DragDropVideoLabel(QLabel):
    """Custom QLabel with drag and drop functionality for videos"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.parent_app = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.is_hovered = False

        # Normal styling
        self.normal_style = f"""
            QLabel {{
                {DEFAULT_FONT}
                font-size: 14px;
                color: {MUTED_TEXT_COLOR};
                background-color: {DARK_BG_COLOR};
                border-radius: 4px;
            }}
        """

        # Mouse hover styling (without drag)
        self.hover_style = f"""
            QLabel {{
                {DEFAULT_FONT}
                font-size: 14px;
                color: {MUTED_TEXT_COLOR};
                background-color: #1a1a1a;
                border-radius: 4px;
            }}
        """

        # Highlight styling for drag hover
        self.highlight_style = f"""
            QLabel {{
                {DEFAULT_FONT}
                font-size: 14px;
                color: #FFFFFF;
                background-color: #1a1a1a;
                border: 2px dashed {ACCENT_COLOR};
                border-radius: 4px;
            }}
        """

        self.setStyleSheet(self.normal_style)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.content_layout = QVBoxLayout(self)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(15)

        self.icon_label = QLabel(self)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setFixedSize(80, 80)
        self.icon_label.setStyleSheet("border: none;")  # Remove borders

        self.text_label = QLabel(f"<span style='color:{ACCENT_COLOR};'>(Upload Video)</span><br>Or select a sample source and press play")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet(f"color: {MUTED_TEXT_COLOR}; border: none;")  # Remove borders

        self.content_layout.addStretch(1)
        self.content_layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(self.text_label, 0, Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addStretch(1)

    def set_parent_app(self, app):
        """Set parent application reference to access video loading methods"""
        self.parent_app = app

    def set_default_content(self):
        """Set the default content with SVG icon and text"""
        # Clear the video pixmap if present
        self.clear()

        if self.is_hovered:
            self.setStyleSheet(self.hover_style)
        else:
            self.setStyleSheet(self.normal_style)

        assets_dir = os.path.join(os.getcwd(), "assets")
        svg_path = os.path.join(assets_dir, "video-upload.svg")

        if os.path.exists(svg_path):
            try:
                svg_renderer = QSvgRenderer(svg_path)
                if svg_renderer.isValid():
                    pixmap = QPixmap(80, 80)
                    pixmap.fill(Qt.GlobalColor.transparent)
                    painter = QPainter(pixmap)
                    svg_renderer.render(painter)
                    painter.end()
                    self.icon_label.setPixmap(pixmap)
                else:
                    # Fallback
                    self.icon_label.setText("📁")
                    self.icon_label.setStyleSheet(f"font-size: 48px; color: {MUTED_TEXT_COLOR}; border: none;")
            except Exception as e:
                # Fallback icon
                self.icon_label.setText("📁")
                self.icon_label.setStyleSheet(f"font-size: 48px; color: {MUTED_TEXT_COLOR}; border: none;")
        else:
            # Fallback icon
            self.icon_label.setText("📁")
            self.icon_label.setStyleSheet(f"font-size: 48px; color: {MUTED_TEXT_COLOR}; border: none;")

        self.icon_label.setVisible(True)
        self.text_label.setVisible(True)

    def enterEvent(self, event):
        """Handle mouse enter events - darken background"""
        if not self.is_hovered:
            self.is_hovered = True
            self.setStyleSheet(self.hover_style)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave events - restore normal background"""
        if self.is_hovered:
            self.is_hovered = False
            self.setStyleSheet(self.normal_style)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle click events to open file dialog"""
        if self.parent_app:
            self.parent_app.open_file_dialog()
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag enter events that contain file URLs"""
        if event.mimeData().hasUrls():
            self.is_hovered = True
            self.setStyleSheet(self.highlight_style)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        """Reset style when drag leaves"""
        self.is_hovered = False
        self.setStyleSheet(self.normal_style)
        super().dragLeaveEvent(event)

    def dragMoveEvent(self, event):
        """Accept drag move events for file URLs"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle file drop events"""
        # Reset style
        self.is_hovered = False
        self.setStyleSheet(self.normal_style)

        if event.mimeData().hasUrls() and self.parent_app is not None:
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()  # Get the first dropped file path
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
                if any(file_path.lower().endswith(ext) for ext in video_extensions):
                    self.parent_app.load_video_from_path(file_path)
                    event.acceptProposedAction()
