import sys
import os
import cv2
import torch
import numpy as np
import urllib.request
import time
from collections import deque

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QFrame, QSizePolicy, QFileDialog, QProgressBar,
                             QDialog, QSplitter, QScrollArea, QGridLayout, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QSize, QThread, pyqtSignal, QRect
from PyQt6.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent, QPainter

import pyqtgraph as pg

# Project modules
from config import * 
from core.threads import ModelDownloadThread, VideoFrameThread, YoloDetectionThread
from ui.widgets import ToggleSwitch, ModernBoxedSlider, DragDropVideoLabel


class CrowdSenseApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CrowdSense")
        self.setMinimumSize(1100, 700)

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {DARK_BG_COLOR};
                color: {TEXT_COLOR};
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(24, 24, 24, 24)
        self.main_layout.setSpacing(16)

        self.export_heatmap_button = None
        self.export_graph_button = None

        self.available_models = available_models

        # Initialize model to YOLOv8n by default
        self.current_model_key = "YOLOv8n (Nano)"
        self.model_path = self.available_models[self.current_model_key]["path"]

        # Directory for storing models
        self.models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Flag to track if model is downloading
        self.model_downloading = False
        self.yolo_ready = False

        # Initialize video capture in a separate thread
        self.cap = None
        self.video_thread = VideoFrameThread()
        self.video_thread.frame_ready.connect(self.process_video_frame)
        self.video_thread.video_ended.connect(self.on_video_ended) # Connect end signal

        # Initialize YOLO detection thread
        self.yolo_thread = YoloDetectionThread(self.model_path)
        self.yolo_thread.detection_ready.connect(self.display_detection_results)
        self.yolo_thread.model_loaded.connect(self.on_model_loaded)

        # Initialize model download thread (will be created when needed)
        self.download_thread = None

        # Frame buffers and detection data
        self.current_frame = None  # Raw current frame
        self.displayed_frame = None  # Processed frame with heatmap (if enabled)
        self.last_detected_boxes = []  # Store the last detected boxes

        # People counting
        self.people_count = 0

        self.smoothing_window_size = 24  # Default window size (used in init)
        self.people_count_history = deque(maxlen=self.smoothing_window_size)
        self.smoothed_people_count = 0

        # Playback state
        self.paused = False
        self.confidence_threshold = 0.4  # Default value

        # Heatmap properties
        self.heatmap_enabled = False
        self.heatmap_opacity = 0.7
        self.heatmap_accumulator = None
        self.aggregate_heatmap_accumulator = None # This will store the aggregate heatmap with no decay
        self.aggregate_frame_count = 0  # Track how many frames contributed to aggregate
        self.heatmap_decay = 0.99
        self.heatmap_blur_size = 21
        self.heatmap_radius = 2
        self.heatmap_intensity = 0.6
        self.heatmap_scale_factor = 0.2
        self.heatmap_neighbor_radius = 4

        # Video timer properties
        self.video_time_ms = 0
        self.last_frame_time = 0
        self.frame_interval = 33  # Default frame interval

        # Crowd threshold parameters
        self.crowd_detection_enabled = False
        self.crowd_size_threshold = 10  # Default threshold for people count
        self.threshold_alert_active = False
        self.threshold_history = []  # Store alert history with timestamps

        self.peak_count = 0
        self.peak_time_ms = 0
        self.offpeak_count = float('inf')  # Start with infinity so any count will be lower
        self.offpeak_time_ms = 0
        self.peak_marker = None
        self.offpeak_marker = None

        self.setup_ui()

    def setup_ui(self):
        self.setup_header()
        self.setup_main_content()

    def setup_header(self):
        header_container = QWidget()
        header_layout = QVBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)

        # Title and subtitle
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)

        title = QLabel("CrowdSense")
        title.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 26px;
            font-weight: bold;
            color: #FFFFFF;
        """)

        subtitle = QLabel("A Real-Time Crowd Monitoring Utility")
        subtitle.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 14px;
            color: {MUTED_TEXT_COLOR};
        """)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        # Controls section
        controls_section = QWidget()
        controls_section_layout = QVBoxLayout(controls_section)
        controls_section_layout.setContentsMargins(0, 0, 0, 0)
        controls_section_layout.setSpacing(8)

        # First row: Model selection and confidence threshold
        self.create_model_selection_row(controls_section_layout)

        # Second row: Video source selection and playback controls
        self.create_source_selection_row(controls_section_layout)

        header_layout.addWidget(title_container)
        header_layout.addWidget(controls_section)

        self.main_layout.addWidget(header_container)

    def create_model_selection_row(self, parent_layout):
        first_row_container = QWidget()
        first_row_layout = QHBoxLayout(first_row_container)
        first_row_layout.setContentsMargins(0, 0, 0, 0)
        first_row_layout.setSpacing(16)

        # Model selection part
        model_container = QWidget()
        model_layout = QHBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)

        model_label = QLabel("Select YOLO Model:")
        model_label.setStyleSheet(SUBHEADER_FONT_STYLE)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.setup_dropdown_style(self.model_combo)

        # Model combo box
        for model_name in self.available_models:
            model_info = self.available_models[model_name]
            display_text = f"{model_name} - {model_info['description']} ({model_info['size']})"
            self.model_combo.addItem(display_text, model_name)

        # Select the default model
        default_index = list(self.available_models.keys()).index(self.current_model_key)
        self.model_combo.setCurrentIndex(default_index)

        # Connect model selection change event
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)

        # Confidence threshold part
        threshold_container = QWidget()
        threshold_layout = QHBoxLayout(threshold_container)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(8)

        threshold_label = QLabel("Set Confidence ≥:")
        threshold_label.setStyleSheet(SUBHEADER_FONT_STYLE)

        self.threshold_slider = ModernBoxedSlider()
        self.threshold_slider.setMinimumWidth(100)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(int(self.confidence_threshold * 100))
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider, 1)

        # Add both containers to the first row
        first_row_layout.addWidget(model_container, 3)
        first_row_layout.addWidget(threshold_container, 1)

        parent_layout.addWidget(first_row_container)

    def create_source_selection_row(self, parent_layout):
        source_container = QWidget()
        source_layout = QHBoxLayout(source_container)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(8)

        source_label = QLabel("Select Sample Source:")
        source_label.setStyleSheet(SUBHEADER_FONT_STYLE)

        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(300)
        self.setup_dropdown_style(self.source_combo)

        self.populate_sources()

        # Control buttons container
        buttons_container = self.create_playback_buttons()

        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_combo, 1)
        source_layout.addWidget(buttons_container)

        parent_layout.addWidget(source_container)

    def create_playback_buttons(self):
        buttons_container = QWidget()
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)

        self.restart_button = QPushButton("↻")
        self.restart_button.setToolTip("Restart Video")
        self.restart_button.setStyleSheet(BUTTON_STYLE)
        self.restart_button.clicked.connect(self.restart_video)
        self.restart_button.setEnabled(False)  # Initially disabled

        self.play_button = QPushButton("▶")
        self.play_button.setToolTip("Start")
        self.play_button.setStyleSheet(BUTTON_STYLE)
        self.play_button.clicked.connect(self.start_video)

        self.pause_button = QPushButton("⏸")
        self.pause_button.setToolTip("Pause")
        self.pause_button.setStyleSheet(BUTTON_STYLE)
        self.pause_button.clicked.connect(self.pause_video)
        self.pause_button.setEnabled(False)

        self.stop_button = QPushButton("⏹")
        self.stop_button.setToolTip("Stop")
        self.stop_button.setStyleSheet(BUTTON_STYLE)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)

        # Add buttons to layout
        buttons_layout.addWidget(self.restart_button)
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.stop_button)

        return buttons_container

    def create_peak_time_widget(self):
        """Create a simplified widget for displaying peak and off-peak times"""
        peak_time_widget = QWidget()
        peak_time_widget.setStyleSheet(f"""
            background-color: {WIDGET_BG_COLOR};
            border-radius: 4px;
        """)

        peak_layout = QVBoxLayout(peak_time_widget)
        peak_layout.setContentsMargins(16, 16, 16, 16)
        peak_layout.setSpacing(12)

        # Header for the section - match styling with other section headers
        section_header = QLabel("Traffic Analysis")
        section_header.setStyleSheet(f"""
            font-family: Arial;
            font-size: 14px;
            color: #CCCCCC;
            border: none;
        """)
        peak_layout.addWidget(section_header)

        # Container for peak and off-peak rows
        rows_container = QWidget()
        rows_container.setStyleSheet("border: none;")
        rows_layout = QVBoxLayout(rows_container)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(8)  # Space between rows

        # Peak time row (horizontal)
        peak_row = QWidget()
        peak_row.setStyleSheet("border: none;")
        peak_row_layout = QHBoxLayout(peak_row)
        peak_row_layout.setContentsMargins(0, 0, 0, 0)
        peak_row_layout.setSpacing(12)  # Space between elements

        # Small colored indicator for peak
        peak_indicator = QLabel()
        peak_indicator.setFixedSize(10, 10)
        peak_indicator.setStyleSheet("""
            background-color: #FF5555;
            border-radius: 5px;
            border: none;
        """)

        peak_label = QLabel("Peak Time:")
        peak_label.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            font-weight: bold;
            color: #CCCCCC;
            border: none;
        """)
        peak_label.setFixedWidth(100)

        self.peak_time_value = QLabel("--:--:--")
        self.peak_time_value.setStyleSheet(f"""
            font-family: Arial;
            font-size: 14px;
            font-weight: bold;
            color: {ACCENT_COLOR};
            border: none;
        """)
        self.peak_time_value.setFixedWidth(60)

        self.peak_count_value = QLabel("(0 people)")
        self.peak_count_value.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #AAAAAA;
            border: none;
        """)

        peak_row_layout.addWidget(peak_indicator)
        peak_row_layout.addWidget(peak_label)
        peak_row_layout.addWidget(self.peak_time_value)
        peak_row_layout.addWidget(self.peak_count_value)
        peak_row_layout.addStretch(1)  # Push everything to the left

        # Off-peak time row (horizontal)
        offpeak_row = QWidget()
        offpeak_row.setStyleSheet("border: none;")
        offpeak_row_layout = QHBoxLayout(offpeak_row)
        offpeak_row_layout.setContentsMargins(0, 0, 0, 0)
        offpeak_row_layout.setSpacing(12)

        # Small colored indicator for off-peak
        offpeak_indicator = QLabel()
        offpeak_indicator.setFixedSize(10, 10)
        offpeak_indicator.setStyleSheet("""
            background-color: #5599FF;
            border-radius: 5px;
            border: none;
        """)

        offpeak_label = QLabel("Off-Peak Time:")
        offpeak_label.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            font-weight: bold;
            color: #CCCCCC;
            border: none;
        """)
        offpeak_label.setFixedWidth(100)

        self.offpeak_time_value = QLabel("--:--:--")
        self.offpeak_time_value.setStyleSheet(f"""
            font-family: Arial;
            font-size: 14px;
            font-weight: bold;
            color: {ACCENT_COLOR};
            border: none;
        """)
        self.offpeak_time_value.setFixedWidth(60)

        self.offpeak_count_value = QLabel("(0 people)")
        self.offpeak_count_value.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #AAAAAA;
            border: none;
        """)

        offpeak_row_layout.addWidget(offpeak_indicator)
        offpeak_row_layout.addWidget(offpeak_label)
        offpeak_row_layout.addWidget(self.offpeak_time_value)
        offpeak_row_layout.addWidget(self.offpeak_count_value)
        offpeak_row_layout.addStretch(1)

        rows_layout.addWidget(peak_row)
        rows_layout.addWidget(offpeak_row)

        peak_layout.addWidget(rows_container)

        return peak_time_widget

    def setup_dropdown_style(self, combobox):
        """Apply consistent styling to dropdown menus"""
        combobox.setStyleSheet(f"""
            QComboBox {{
                background-color: {DROPDOWN_BG_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                padding: 6px 12px;
                {DEFAULT_FONT}
                font-size: 14px;
                min-height: 20px;
            }}

            QComboBox:hover {{
                border: 1px solid {ACCENT_COLOR};
            }}

            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 20px;
                border-left: none;
                padding-right: 5px;
            }}

            QComboBox QAbstractItemView {{
                background-color: {PANEL_BG_COLOR};
                border: 1px solid {BORDER_COLOR};
                selection-background-color: {ACCENT_COLOR};
            }}

            QComboBox QAbstractItemView::item {{
                padding: 6px 12px;
                min-height: 20px;
            }}

            QComboBox QAbstractItemView::item:hover {{
                background-color: {BORDER_COLOR};
            }}
        """)

    def setup_main_content(self):
        # Create main content area with video and stats side by side
        main_content = QWidget()
        main_content_layout = QVBoxLayout(main_content)
        main_content_layout.setContentsMargins(0, 0, 0, 0)
        main_content_layout.setSpacing(16)

        # Create a splitter for resizable sections
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)  # Prevent sections from being collapsed
        self.main_splitter.setHandleWidth(4)
        self.main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: transparent; /* Make handle invisible */
                margin: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {ACCENT_COLOR};
            }}
        """)
        self.main_splitter.setStretchFactor(0, 1)  # Video container can stretch
        self.main_splitter.setStretchFactor(1, 0)  # Metrics container has fixed width preference
        self.main_splitter.splitterMoved.connect(self.on_splitter_moved)

        video_container = QWidget()
        metrics_container = QWidget()

        metrics_container.setMinimumWidth(350)

        self.setup_video_output(video_container)
        self.setup_metrics_panel(metrics_container)

        self.main_splitter.addWidget(video_container)
        self.main_splitter.addWidget(metrics_container)

        self.main_splitter.setSizes([700, 300])

        main_content_layout.addWidget(self.main_splitter)

        # Add main content to main layout with stretch factor
        self.main_layout.addWidget(main_content, 1)

    def on_splitter_moved(self, pos, index):
        """Handle splitter movement to update video frame"""
        # Force a resize event to update the video display
        if self.displayed_frame is not None and self.paused:
            rgb_frame = cv2.cvtColor(self.displayed_frame, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb_frame)

    def setup_video_output(self, parent_widget):
        # Apply styling directly to the parent widget
        parent_widget.setStyleSheet(f"""
            background-color: {PANEL_BG_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
        """)

        output_layout = QVBoxLayout(parent_widget)
        output_layout.setContentsMargins(16, 16, 16, 16)
        output_layout.setSpacing(12)

        header_container = QWidget()
        header_container.setStyleSheet("border: none;")
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        output_header = QLabel("Video Feed with Detection:")
        output_header.setStyleSheet(HEADER_FONT_STYLE)

        # Add toggle switch for heatmap
        heatmap_container = QWidget()
        heatmap_container.setStyleSheet("border: none;")
        heatmap_layout = QHBoxLayout(heatmap_container)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_layout.setSpacing(8)

        heatmap_label = QLabel("Show Density Heatmap:")
        heatmap_label.setStyleSheet(SUBHEADER_FONT_STYLE)

        self.heatmap_toggle = ToggleSwitch()
        self.heatmap_toggle.toggled.connect(self.on_heatmap_toggled)
        self.heatmap_toggle.setEnabled(False)  # Initially disabled since no video is playing

        heatmap_layout.addWidget(heatmap_label)
        heatmap_layout.addWidget(self.heatmap_toggle)

        header_layout.addWidget(output_header)
        header_layout.addStretch(1)
        header_layout.addWidget(heatmap_container)

        # Timer display
        timer_container = QWidget()
        timer_container.setStyleSheet("border: none;")
        timer_layout = QHBoxLayout(timer_container)
        timer_layout.setContentsMargins(0, 0, 0, 0)
        timer_layout.setSpacing(8)

        timer_label = QLabel("Elapsed Time:")
        timer_label.setStyleSheet(SUBHEADER_FONT_STYLE)

        self.timer_display = QLabel("00:00:00:000")
        self.timer_display.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 14px;
            font-weight: bold;
            color: {ACCENT_COLOR};
            border: none;
            min-width: 80px;
        """)

        # Container for timer and end indicator
        timer_display_container = QWidget()
        timer_display_container.setStyleSheet("border: none;")
        timer_display_layout = QHBoxLayout(timer_display_container)
        timer_display_layout.setContentsMargins(0, 0, 0, 0)
        timer_display_layout.setSpacing(4)

        timer_display_layout.addWidget(self.timer_display)

        self.end_playback_label = QLabel("(End of playback reached)")
        self.end_playback_label.setStyleSheet("color: #999999; font-size: 12px;") # Changed to grey, removed italic
        self.end_playback_label.setVisible(False)  # Initially hidden

        timer_display_layout.addWidget(self.end_playback_label)
        timer_display_layout.addStretch(1)

        timer_layout.addWidget(timer_label)

        timer_layout.addWidget(timer_display_container)
        timer_layout.addStretch(1)

        # Model loading indicator
        self.model_status = QLabel("YOLO Model: Not Loaded")
        self.model_status.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 12px;
            color: {MUTED_TEXT_COLOR};
            border: none;
        """)

        # Model loading progress bar
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setValue(0)
        self.model_progress.setVisible(False)
        self.model_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                background-color: {PANEL_BG_COLOR};
                height: 6px;
                text-align: center;
                color: {TEXT_COLOR};
            }}

            QProgressBar::chunk {{
                background-color: {ACCENT_COLOR};
            }}
        """)

        # Video container
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        video_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Video label with drag and drop
        self.video_label = DragDropVideoLabel()
        self.video_label.set_parent_app(self)
        self.video_label.set_default_content()
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumWidth(100)  # Set a very small minimum width

        video_layout.addWidget(self.video_label)

        # Add components to output layout
        output_layout.addWidget(header_container)
        output_layout.addWidget(timer_container)
        output_layout.addWidget(self.model_status)
        output_layout.addWidget(self.model_progress)
        output_layout.addWidget(video_container, 1)  # Give video container stretch priority
        output_layout.addSpacing(8)

        # Export Heatmap button container
        export_container = QWidget()
        export_container.setStyleSheet("border: none; margin: 0;")  # Remove border
        export_layout = QHBoxLayout(export_container)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(10)

        # Heatmap export button
        self.export_heatmap_button = QPushButton("Export Heatmap")
        self.export_heatmap_button.setStyleSheet(EXPORT_BUTTON_STYLE)
        self.export_heatmap_button.setFixedWidth(150)
        self.export_heatmap_button.setEnabled(False)  # Initially disabled
        self.export_heatmap_button.clicked.connect(self.export_heatmap)

        export_layout.addWidget(self.export_heatmap_button)
        export_layout.addStretch(1)

        output_layout.addWidget(export_container)

    def setup_metrics_panel(self, parent_widget):
        """Setup the metrics panel with improved vertical space handling"""
        parent_widget.setStyleSheet(f"""
            background-color: {PANEL_BG_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
        """)

        main_layout = QVBoxLayout(parent_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area to handle limited vertical space
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: {PANEL_BG_COLOR};
                width: 8px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR};
                min-height: 20px;
                border-radius: 4px; /* Adjusted radius */
            }}
            QScrollBar::handle:vertical:hover {{
                background: {ACCENT_COLOR};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none;
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)

        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: transparent; border: none; border-top: 1px solid {BORDER_COLOR};")
        metrics_layout = QVBoxLayout(scroll_content)
        metrics_layout.setContentsMargins(16, 16, 16, 16)
        metrics_layout.setSpacing(16)

        metrics_header = QLabel("Detection Metrics:")
        metrics_header.setStyleSheet(HEADER_FONT_STYLE)

        people_count_widget = self.create_people_count_widget()
        crowd_detection_widget = self.create_crowd_detection_widget()
        people_graph_widget = self.create_people_graph_widget()
        peak_time_widget = self.create_peak_time_widget()

        metrics_layout.addWidget(metrics_header)
        metrics_layout.addWidget(people_count_widget)
        metrics_layout.addWidget(crowd_detection_widget)
        metrics_layout.addWidget(people_graph_widget, 1)  # Graph should stretch
        metrics_layout.addWidget(peak_time_widget)
        metrics_layout.addStretch(1)

        scroll_area.setWidget(scroll_content)

        export_container = QWidget()
        export_container.setStyleSheet(f"background-color: transparent; border: none; border-top: 1px solid {BORDER_COLOR};")
        export_layout = QVBoxLayout(export_container)
        export_layout.setContentsMargins(16, 8, 16, 16)
        export_layout.setSpacing(0)

        self.export_graph_button = QPushButton("Export People Count Graph")
        self.export_graph_button.setStyleSheet(EXPORT_BUTTON_STYLE)
        self.export_graph_button.setFixedWidth(220)
        self.export_graph_button.setEnabled(False)
        self.export_graph_button.clicked.connect(self.export_count_graph)

        button_container = QWidget()
        button_container.setStyleSheet("background-color: transparent; border: none;")
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)
        button_layout.addWidget(self.export_graph_button)
        button_layout.addStretch(1)

        export_layout.addWidget(button_container)

        main_layout.addWidget(scroll_area, 1)
        main_layout.addWidget(export_container)

    def create_people_count_widget(self):
        """Create more compact people count widget with no border"""
        people_count_widget = QWidget()
        people_count_widget.setStyleSheet(f"""
            background-color: {WIDGET_BG_COLOR};
            border-radius: 4px;
        """)

        people_count_layout = QHBoxLayout(people_count_widget)
        people_count_layout.setContentsMargins(12, 10, 12, 10)
        people_count_layout.setSpacing(10)

        people_count_header = QLabel("People Detected")
        people_count_header.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #CCCCCC;
            border: none;
        """)

        self.people_count_value = QLabel("0")
        self.people_count_value.setStyleSheet(LARGE_VALUE_FONT_STYLE)
        self.people_count_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.people_count_value.setMinimumWidth(60)

        people_count_layout.addWidget(people_count_header)
        people_count_layout.addStretch(1)
        people_count_layout.addWidget(self.people_count_value)

        people_count_widget.setFixedHeight(60)

        return people_count_widget

    def create_people_graph_widget(self):
        self.setup_people_count_graph()

        people_graph_widget = QWidget()
        people_graph_widget.setStyleSheet(f"""
            background-color: {WIDGET_BG_COLOR};
            border-radius: 4px;
        """)

        people_graph_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding  # Allow vertical expansion
        )

        people_graph_layout = QVBoxLayout(people_graph_widget)
        people_graph_layout.setContentsMargins(16, 16, 16, 16)
        people_graph_layout.setSpacing(8)

        people_graph_header = QLabel("People Count Over Time")
        people_graph_header.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 14px;
            color: #CCCCCC;
            border: none;
        """)

        people_graph_layout.addWidget(people_graph_header)
        people_graph_layout.addSpacing(8)
        people_graph_layout.addWidget(self.people_graph_plot_widget, 1)

        return people_graph_widget

    def setup_people_count_graph(self):
        """Setup the real-time people count graph with a modern look"""
        self.people_data = []
        self.time_data = []
        self.people_graph_plot_widget = pg.PlotWidget()

        self.people_graph_plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        self.people_graph_plot_widget.setMinimumHeight(150)
        self.people_graph_plot_widget.setBackground(WIDGET_BG_COLOR)
        self.people_graph_plot_widget.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.people_graph_plot_widget.setContentsMargins(0, 0, 0, 0)
        self.people_graph_plot_widget.getPlotItem().getViewBox().setBorder(pen=None)
        self.people_graph_plot_widget.setFrameShape(QFrame.Shape.NoFrame)

        self.people_graph_line = self.people_graph_plot_widget.plot(
            [], [],
            pen=pg.mkPen(color=ACCENT_COLOR, width=3),
            symbolBrush=pg.mkBrush(LIGHTER_ACCENT_COLOR),
            symbolPen=pg.mkPen(LIGHTER_ACCENT_COLOR),
            symbolSize=4,
            symbol='o'
        )

        axis_color = '#888888'
        self.people_graph_plot_widget.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color=axis_color, width=1))
        self.people_graph_plot_widget.getPlotItem().getAxis('left').setPen(pg.mkPen(color=axis_color, width=1))

        self.people_graph_plot_widget.setLabel('left', 'People Count', color='#CCCCCC')
        self.people_graph_plot_widget.setLabel('bottom', 'Time (s)', color='#CCCCCC')

        self.people_graph_plot_widget.showGrid(x=True, y=True, alpha=0.2)

        self.threshold_line = None
        self.alert_segment = None


    def create_crowd_detection_widget(self):
        """Create widget for crowd threshold detection and alerts"""
        crowd_widget = QWidget()
        crowd_widget.setStyleSheet(f"""
            background-color: {WIDGET_BG_COLOR};
            border-radius: 4px;
        """)

        crowd_layout = QVBoxLayout(crowd_widget)
        crowd_layout.setContentsMargins(12, 12, 12, 12)
        crowd_layout.setSpacing(8)

        header_container = QWidget()
        header_container.setStyleSheet("background-color: transparent; border: none;")
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        crowd_header = QLabel("Crowd Threshold Alerts")
        crowd_header.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #CCCCCC;
            border: none;
        """)

        self.crowd_toggle = ToggleSwitch()
        self.crowd_toggle.toggled.connect(self.on_crowd_detection_toggled)
        self.crowd_toggle.setEnabled(False)

        header_layout.addWidget(crowd_header)
        header_layout.addStretch(1)
        header_layout.addWidget(self.crowd_toggle)

        self.crowd_settings_container = QWidget()
        self.crowd_settings_container.setStyleSheet("background-color: transparent; border: none;")
        settings_layout = QVBoxLayout(self.crowd_settings_container)
        settings_layout.setContentsMargins(0, 8, 0, 0)
        settings_layout.setSpacing(12)

        # Initially hide the settings
        self.crowd_settings_container.setVisible(False)

        threshold_container = QWidget()
        threshold_container.setStyleSheet("background-color: transparent; border: none;")
        threshold_layout = QHBoxLayout(threshold_container)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(8)

        threshold_label = QLabel("People Threshold:")
        threshold_label.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #E0E0E0;
            border: none;
        """)

        self.people_threshold_slider = ModernBoxedSlider(integer_display=True)
        self.people_threshold_slider.setRange(3, 50)  # Adjustable range for people count
        self.people_threshold_slider.setValue(self.crowd_size_threshold)
        self.people_threshold_slider.valueChanged.connect(self.on_crowd_size_threshold_changed)

        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.people_threshold_slider, 1)

        smoothing_container = QWidget()
        smoothing_container.setStyleSheet("background-color: transparent; border: none;")
        smoothing_layout = QHBoxLayout(smoothing_container)
        smoothing_layout.setContentsMargins(0, 0, 0, 0)
        smoothing_layout.setSpacing(8)

        smoothing_label = QLabel("Smoothing Window:")
        smoothing_label.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #E0E0E0;
            border: none;
        """)

        self.smoothing_slider = ModernBoxedSlider(integer_display=True)
        self.smoothing_slider.setRange(1, 60)
        self.smoothing_slider.setValue(self.smoothing_window_size)
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_window_changed)

        smoothing_layout.addWidget(smoothing_label)
        smoothing_layout.addWidget(self.smoothing_slider, 1)

        self.alert_container = QWidget()
        self.alert_container.setStyleSheet(f"""
            background-color: #2A2A2A;
            border-radius: 4px;
            border: 1px solid {BORDER_COLOR};
        """)

        alert_layout = QHBoxLayout(self.alert_container)
        alert_layout.setContentsMargins(10, 8, 10, 8)
        alert_layout.setSpacing(8)

        self.alert_icon = QLabel("🔔")
        self.alert_icon.setStyleSheet("""
            font-family: Arial;
            font-size: 16px;
            color: #555555;
            border: none;
        """)

        self.alert_text = QLabel("People count is normal")
        self.alert_text.setStyleSheet("""
            font-family: Arial;
            font-size: 12px;
            color: #AAAAAA;
            border: none;
        """)

        alert_layout.addWidget(self.alert_icon)
        alert_layout.addWidget(self.alert_text, 1)

        settings_layout.addWidget(threshold_container)
        settings_layout.addWidget(smoothing_container)
        settings_layout.addWidget(self.alert_container)

        crowd_layout.addWidget(header_container)
        crowd_layout.addWidget(self.crowd_settings_container)

        return crowd_widget

    def on_crowd_detection_toggled(self, enabled):
        """Handle crowd detection toggle switch changes"""
        self.crowd_detection_enabled = enabled

        # Make sure that toggle visual state matches functionality
        self.crowd_toggle.setChecked(enabled)

        self.crowd_settings_container.setVisible(enabled)

        if not enabled:
            self.update_crowd_alert_status(False)
            self.threshold_alert_active = False
            # Remove threshold line from graph
            if self.threshold_line is not None:
                self.people_graph_plot_widget.removeItem(self.threshold_line)
                self.threshold_line = None
            if self.alert_segment is not None:
                 self.people_graph_plot_widget.removeItem(self.alert_segment)
                 self.alert_segment = None
        else:
             # Ensure threshold line is added or updated if video is running
             if self.cap is not None and self.cap.isOpened() and not self.paused:
                  self.update_people_graph(self.smoothed_people_count)


    def on_crowd_size_threshold_changed(self, value):
        """Handle crowd size threshold slider change"""
        self.crowd_size_threshold = value
        self.update_crowd_alert_status(False)
        
        # Update threshold line on graph if enabled and running
        if self.crowd_detection_enabled and self.cap is not None and self.cap.isOpened() and not self.paused:
             self.update_people_graph(self.smoothed_people_count)


    def on_smoothing_window_changed(self, value):
        """Handle smoothing window size slider change"""
        current_history = list(self.people_count_history)
        self.smoothing_window_size = value
        self.people_count_history = deque(maxlen=value)

        # Add back the existing history
        for count in current_history[-value:]:
            self.people_count_history.append(count)

        # Recalculate smoothed count
        if len(self.people_count_history) > 0:
            self.smoothed_people_count = round(np.mean(self.people_count_history))
        else:
             self.smoothed_people_count = 0 # Default to 0 if history is empty

        # Update display immediately ONLY if video is NOT running/paused.
        # If running, display_detection_results will handle the update.
        if self.cap is None or not self.cap.isOpened():
             self.people_count_value.setText(str(self.smoothed_people_count))


        # Re-check threshold with new smoothed count if detection is enabled
        if self.crowd_detection_enabled:
            # Use the newly calculated smoothed count
            self.check_threshold_crossing(self.current_frame) # Pass frame if available

    def format_time_for_filename(time_ms):
        """Format time in milliseconds to a string suitable for filenames"""
        # Calculate hours, minutes, seconds
        total_seconds = time_ms // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = time_ms % 1000

        return f"{hours:02d}h{minutes:02d}m{seconds:02d}s{milliseconds:03d}ms"

    def update_crowd_alert_status(self, alert_active, count=0):
        """Update the crowd alert status indicator"""
        if self.threshold_alert_active == alert_active:
             return

        self.threshold_alert_active = alert_active

        if alert_active:
            self.alert_container.setStyleSheet("""
                background-color: #4e1c1c;
                border-radius: 4px;
                border: 1px solid #cc3232; /* Light red border */
            """)
            self.alert_icon.setStyleSheet("""
                font-family: Arial;
                font-size: 16px;
                color: #ff5555;
                border: none;
            """)
            self.alert_text.setStyleSheet("""
                font-family: Arial;
                font-size: 12px;
                color: #ff9999;
                border: none;
            """)

            self.alert_text.setText(f"ALERT! {count} people detected (threshold: {self.crowd_size_threshold})")

            # Also update people count display with alert styling
            if hasattr(self, 'people_count_value'):
                # Use LARGE_VALUE_FONT_STYLE but change color
                alert_font_style = LARGE_VALUE_FONT_STYLE.replace(f"color: {ACCENT_COLOR}", "color: #ff5555")
                self.people_count_value.setStyleSheet(alert_font_style)

            current_time_str = format_time_for_filename(self.video_time_ms)
            alert_record = {
                'timestamp': current_time_str,
                'count': count,
                'threshold': self.crowd_size_threshold
            }
            self.threshold_history.append(alert_record)

        else:
            self.alert_container.setStyleSheet(f"""
                background-color: #2A2A2A;
                border-radius: 4px;
                border: 1px solid {BORDER_COLOR}; /* Grey border */
            """)
            self.alert_icon.setStyleSheet("""
                font-family: Arial;
                font-size: 16px;
                color: #555555;
                border: none;
            """)
            self.alert_text.setStyleSheet("""
                font-family: Arial;
                font-size: 12px;
                color: #AAAAAA;
                border: none;
            """)

            self.alert_text.setText("People count is normal")

            if hasattr(self, 'people_count_value'):
                self.people_count_value.setStyleSheet(LARGE_VALUE_FONT_STYLE) # Use defined large style


    def restart_video(self):
        """Restart the current video from the beginning in a thread-safe way"""
        if self.cap is None or not self.cap.isOpened():
            return

        video_was_at_end = self.end_playback_label.isVisible()

        was_paused = self.paused

        self.end_playback_label.setVisible(False)

        # Always pause video thread first to prevent concurrent access
        self.paused = True
        self.video_thread.pause(True)

        # Stop the video thread completely to ensure we have exclusive access to cap
        video_was_running = self.video_thread.isRunning()
        if video_was_running:
            self.video_thread.stop()
            self.video_thread.wait()

        # Create a small delay to ensure thread has stopped cleanly
        QApplication.processEvents()
        # time.sleep(0.1) # QThread.wait() should be sufficient

        # Reset video position to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Reset timer
        self.video_time_ms = 0
        self.last_frame_time = time.time() # Reset last frame time as well
        self.update_timer_display()

        # Reset graph data and visual elements
        self.people_data.clear()
        self.time_data.clear()
        self.people_graph_line.setData([], []) # Clear graph line data

        # Reset heatmap accumulators
        heatmap_was_enabled = self.heatmap_enabled
        self.heatmap_accumulator = None
        self.aggregate_heatmap_accumulator = None
        self.aggregate_frame_count = 0


        # Reset threshold alert state and graph elements
        self.threshold_alert_active = False
        self.update_crowd_alert_status(False)
        if self.threshold_line is not None:
             self.people_graph_plot_widget.removeItem(self.threshold_line)
             self.threshold_line = None
        if self.alert_segment is not None:
             self.people_graph_plot_widget.removeItem(self.alert_segment)
             self.alert_segment = None

        # Reset peak tracking and markers
        self.peak_count = 0
        self.peak_time_ms = 0
        self.offpeak_count = float('inf')
        self.offpeak_time_ms = 0
        self.peak_time_value.setText("--:--:--")
        self.peak_count_value.setText("(0 people)")
        self.offpeak_time_value.setText("--:--:--")
        self.offpeak_count_value.setText("(0 people)")

        if self.peak_marker is not None:
            self.people_graph_plot_widget.removeItem(self.peak_marker)
            self.peak_marker = None

        if self.offpeak_marker is not None:
            self.people_graph_plot_widget.removeItem(self.offpeak_marker)
            self.offpeak_marker = None

        # Manually read first frame (thread-safe now that video thread is stopped)
        ret, first_frame = self.cap.read()
        display_frame_processed = None
        if ret:
            # Store the raw frame
            self.current_frame = first_frame.copy()

            # Process the first frame directly for immediate display
            if self.yolo_ready and self.yolo_thread.model:
                try:
                     results = self.yolo_thread.model(first_frame, classes=0)
                     boxes = []
                     people_count = 0 # Count for the first frame
                     for result in results:
                          result_boxes = result.boxes
                          for box in result_boxes:
                               x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                               confidence = float(box.conf[0])
                               if confidence > self.confidence_threshold:
                                    boxes.append((x1, y1, x2, y2))
                                    people_count += 1
                                    # Optionally draw boxes on first_frame for immediate feedback
                                    cv2.rectangle(first_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    conf_text = f"{confidence:.2f}"
                                    cv2.putText(first_frame, conf_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                     # Store these boxes for potential heatmap use
                     self.last_detected_boxes = boxes
                     # Update count display for the first frame
                     self.people_count_history.clear() # Reset history
                     self.people_count_history.append(people_count)
                     self.smoothed_people_count = people_count
                     self.people_count_value.setText(str(self.smoothed_people_count))
                     # Update graph for the first point at time 0
                     self.update_people_graph(self.smoothed_people_count)
                     # Check threshold for the first frame
                     if self.crowd_detection_enabled:
                          self.check_threshold_crossing(first_frame) # Pass the frame

                     # Apply heatmap if enabled
                     display_frame_processed = self.process_frame_with_heatmap(first_frame, boxes)

                except Exception as e:
                     print(f"Error processing first frame on restart: {e}")
                     # Fallback: use the raw frame without detections/heatmap
                     display_frame_processed = first_frame.copy()
            else:
                 # YOLO not ready, use the raw frame
                 display_frame_processed = first_frame.copy()
                 self.last_detected_boxes = [] # No boxes detected

            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(display_frame_processed, cv2.COLOR_BGR2RGB)
            # Store the final displayed frame
            self.displayed_frame = display_frame_processed

            # Display the frame
            self.display_frame(rgb_frame)

            # Reset video position again after reading the first frame manually
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
             # If reading the first frame fails, stop everything
             self.stop_video()
             return


        # Restart video thread if it was running before OR if it was at the end
        if video_was_running or video_was_at_end:
            self.video_thread.set_capture(self.cap) # Ensure it has the correct cap object
            self.video_thread.start()

            # Determine if we should be paused or playing
            # If video was at the end, always start playing regardless of previous state
            should_be_paused = was_paused and not video_was_at_end

            self.paused = should_be_paused
            self.video_thread.pause(should_be_paused)

            # Update button states based on the final pause state
            self.play_button.setEnabled(should_be_paused)
            self.pause_button.setEnabled(not should_be_paused)
            self.stop_button.setEnabled(True) # Video is active again
            self.restart_button.setEnabled(True)
        else:
             # If the video wasn't running and wasn't at the end (e.g., stopped state),
             # keep it paused after showing the first frame.
             self.paused = True
             self.video_thread.pause(True)
             # Update buttons for paused state
             self.play_button.setEnabled(True)
             self.pause_button.setEnabled(False)
             self.stop_button.setEnabled(True)
             self.restart_button.setEnabled(True)


    def populate_sources(self):
        # Try to find sources directory
        sources_dir = os.path.join(os.getcwd(), "sources")

        self.source_combo.clear() # Clear existing items first

        if os.path.exists(sources_dir) and os.path.isdir(sources_dir):
            # List all files in the sources directory
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            found_videos = False
            for file in sorted(os.listdir(sources_dir)): # Sort files alphabetically
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    self.source_combo.addItem(file, os.path.join(sources_dir, file))
                    found_videos = True

            # If no videos found, add a placeholder
            if not found_videos:
                self.source_combo.addItem("No videos found in 'sources'", "")
                self.source_combo.setEnabled(False) # Disable if no videos
            else:
                 self.source_combo.setEnabled(True)
        else:
            # If sources directory doesn't exist, add a placeholder and disable
            self.source_combo.addItem("'sources' directory not found", "")
            self.source_combo.setEnabled(False)


    def process_frame_with_heatmap(self, frame, boxes):
        """Process a frame, applying heatmap overlay if enabled"""
        # Create a copy of the frame for display
        # Important: Ensure frame is not None
        if frame is None:
             return None

        display_frame = frame.copy()

        # Apply heatmap overlay if enabled
        if self.heatmap_enabled:
            # Update heatmap with new positions - this adds to the accumulator
            heatmap = self.update_heatmap(display_frame, boxes)

            if heatmap is not None and np.max(heatmap) > 0:
                 # Normalize heatmap for visualization (0 to 1)
                 heatmap_norm = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                 # Ensure minimum value for blue background in low activity areas
                 # Applying this after normalization might be more consistent
                 viz_heatmap = np.maximum(heatmap_norm, 0.1)

                 # Convert to 8-bit for colormap
                 viz_heatmap_8bit = (viz_heatmap * 255).astype(np.uint8)

                 # Apply JET colormap to get blue->green->red gradient
                 heatmap_colored = cv2.applyColorMap(viz_heatmap_8bit, cv2.COLORMAP_JET)

                 # Darken the original frame to make heatmap more visible
                 darkened_frame = cv2.addWeighted(display_frame, 0.4, np.zeros_like(display_frame), 0.6, 0)

                 # Blend the heatmap with the darkened original frame using heatmap_opacity
                 display_frame = cv2.addWeighted(heatmap_colored, self.heatmap_opacity, darkened_frame, 1 - self.heatmap_opacity, 0)

                 # Add grid lines for better visualization
                 h, w = display_frame.shape[:2]
                 grid_spacing = 50

                 # Draw vertical grid lines
                 for x in range(0, w, grid_spacing):
                      cv2.line(display_frame, (x, 0), (x, h), GRID_COLOR, 1)

                 # Draw horizontal grid lines
                 for y in range(0, h, grid_spacing):
                      cv2.line(display_frame, (0, y), (w, y), GRID_COLOR, 1)
            # If heatmap is enabled but no heatmap data (e.g., no detections yet),
            # still return the original frame copy
            # else: # No heatmap data, return original frame copy
                 # pass (display_frame is already frame.copy())

        # Add threshold alert visualization if active (applied AFTER heatmap)
        if self.crowd_detection_enabled and self.threshold_alert_active:
            # Add red border to indicate alert
            h, w = display_frame.shape[:2]
            border_thickness = 8
            border_color_bgr = (0, 0, 200) # Red in BGR
            # Top border
            display_frame[0:border_thickness, 0:w] = border_color_bgr
            # Bottom border
            display_frame[h-border_thickness:h, 0:w] = border_color_bgr
            # Left border
            display_frame[0:h, 0:border_thickness] = border_color_bgr
            # Right border
            display_frame[0:h, w-border_thickness:w] = border_color_bgr

            # Add alert text
            alert_text = f"ALERT! {self.smoothed_people_count} people (threshold: {self.crowd_size_threshold})"
            # Position text inside the border
            text_x = border_thickness + 10
            text_y = border_thickness + 30
            cv2.putText(display_frame, alert_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Red text

        return display_frame


    def update_heatmap(self, frame, boxes):
        """Update the heatmap accumulator with new people positions using a low-resolution approach"""
        # Ensure frame is valid
        if frame is None:
             return None

        h, w = frame.shape[:2]

        # Use the class property for scale factor
        scale_factor = self.heatmap_scale_factor
        low_h, low_w = int(h * scale_factor), int(w * scale_factor)

        # Handle potential division by zero if frame dimensions are tiny
        if low_h <= 0 or low_w <= 0:
             return None # Cannot create heatmap for zero-sized dimensions

        # Check if the frame resolution has changed
        resolution_changed = False
        if self.heatmap_accumulator is not None:
            current_low_h, current_low_w = self.heatmap_accumulator.shape
            if current_low_h != low_h or current_low_w != low_w:
                 resolution_changed = True

        # Initialize low-resolution heatmap accumulator if not exists or resolution changed
        if self.heatmap_accumulator is None or resolution_changed:
            self.heatmap_accumulator = np.zeros((low_h, low_w), dtype=np.float32)

        # Initialize aggregate heatmap accumulator if not exists or resolution changed
        if self.aggregate_heatmap_accumulator is None or resolution_changed:
            self.aggregate_heatmap_accumulator = np.zeros((low_h, low_w), dtype=np.float32)
            self.aggregate_frame_count = 0 # Reset count if resolution changed


        # Apply decay to existing heatmap (only the regular one, not the aggregate)
        self.heatmap_accumulator *= self.heatmap_decay

        # Create a new low-resolution heatmap for current positions
        current_heatmap = np.zeros((low_h, low_w), dtype=np.float32)

        # Add detected people positions to current heatmap
        detections_in_frame = False
        for box in boxes:
            # Scale down coordinates to low resolution
            x1, y1, x2, y2 = box

            # Use bottom center of bounding box (feet position)
            foot_x = int((x1 + x2) / 2 * scale_factor)
            foot_y = int(y2 * scale_factor)

            # Ensure coordinates are within frame boundaries
            foot_x = max(0, min(low_w - 1, foot_x))
            foot_y = max(0, min(low_h - 1, foot_y))

            # Add a point with intensity 1.0
            current_heatmap[foot_y, foot_x] = 1.0
            detections_in_frame = True

            # Add intensity to surrounding pixels (optional, based on original code)
            radius = self.heatmap_neighbor_radius
            y_min = max(0, foot_y - radius)
            y_max = min(low_h - 1, foot_y + radius)
            x_min = max(0, foot_x - radius)
            x_max = min(low_w - 1, foot_x + radius)

            # Efficiently update neighbors using slicing and distance calculation if needed,
            # or simpler loop as in original:
            for y in range(y_min, y_max + 1):
                 for x in range(x_min, x_max + 1):
                      if x == foot_x and y == foot_y:
                           continue
                      dist_sq = (x - foot_x)**2 + (y - foot_y)**2
                      if dist_sq <= radius**2:
                           # Example: linear falloff based on distance
                           dist = np.sqrt(dist_sq)
                           intensity = max(0.0, 1.0 - (dist / radius)) * 0.7 # Weighted intensity
                           current_heatmap[y, x] = max(current_heatmap[y, x], intensity)



        # Apply Gaussian blur passes only if there were detections
        if detections_in_frame:
             # Multiple blur passes as in original logic
             current_heatmap = cv2.GaussianBlur(current_heatmap, (7, 7), 0)
             current_heatmap = cv2.GaussianBlur(current_heatmap, (17, 17), 0)
             current_heatmap = cv2.GaussianBlur(current_heatmap, (31, 31), 0)

             # Normalize the current heatmap before adding to accumulators
             max_val = np.max(current_heatmap)
             if max_val > 0:
                  current_heatmap /= max_val
             else:
                  # Avoid division by zero if blur somehow resulted in all zeros
                  current_heatmap.fill(0)

        # Add current heatmap to accumulator with appropriate intensity
        self.heatmap_accumulator += current_heatmap * self.heatmap_intensity

        # Add to aggregate heatmap accumulator without decay only if detections occurred
        if detections_in_frame:
            self.aggregate_heatmap_accumulator += current_heatmap # Add the normalized, blurred heatmap
            self.aggregate_frame_count += 1

        # Cap the maximum value of the decaying accumulator to prevent excessive brightness
        # This normalization should happen *after* adding the current frame's intensity
        max_accum_val = np.max(self.heatmap_accumulator)
        if max_accum_val > 1.0:
            self.heatmap_accumulator /= max_accum_val
        # Ensure accumulator values stay non-negative
        self.heatmap_accumulator = np.maximum(self.heatmap_accumulator, 0.0)


        # Upsample back to original resolution for display
        # Use INTER_LINEAR for smoother results
        return cv2.resize(self.heatmap_accumulator, (w, h), interpolation=cv2.INTER_LINEAR)


    def update_people_graph(self, count):
        """Update the people count graph with new data and threshold line"""
        # Only update when playing video (not paused or stopped)
        if self.cap is None or not self.cap.isOpened() or self.paused:
            return

        # Use video time in seconds for x-axis
        current_time_sec = self.video_time_ms / 1000.0

        # Add current time and count to data (only if time has advanced)
        if not self.time_data or current_time_sec > self.time_data[-1]:
            self.time_data.append(current_time_sec)
            self.people_data.append(count)
        elif self.time_data: # If time hasn't advanced, update the last count value
             self.people_data[-1] = count


        # Update the graph line data
        self.people_graph_line.setData(self.time_data, self.people_data)

        # Add or update threshold line if crowd detection is enabled
        if self.crowd_detection_enabled:
            # Determine the extent of the threshold line based on current time
            max_time_display = max(current_time_sec + 10, 60) # Extend slightly beyond current time

            if self.threshold_line is None:
                # Create a new threshold line - dashed horizontal line
                pen = pg.mkPen(color='r', width=1, style=Qt.PenStyle.DashLine)
                self.threshold_line = self.people_graph_plot_widget.plot(
                    [0, max_time_display],
                    [self.crowd_size_threshold, self.crowd_size_threshold],
                    pen=pen,
                    name='Threshold'
                )
            else:
                # Update existing threshold line's data and range
                self.threshold_line.setData(
                    [0, max_time_display],
                    [self.crowd_size_threshold, self.crowd_size_threshold]
                )

            # Color the graph segment if alert is active
            if self.threshold_alert_active and len(self.time_data) >= 2:
                 red_pen = pg.mkPen(color='r', width=3)
                 if self.alert_segment is None:
                      self.alert_segment = self.people_graph_plot_widget.plot(
                           self.time_data[-2:], self.people_data[-2:], pen=red_pen
                      )
                 else:
                      self.alert_segment.setData(self.time_data[-2:], self.people_data[-2:])
            elif not self.threshold_alert_active and self.alert_segment is not None:
                 # Remove the alert segment if the alert is no longer active
                 self.people_graph_plot_widget.removeItem(self.alert_segment)
                 self.alert_segment = None

        # Always show the full time range from 0 to current time + padding
        if current_time_sec >= 0: # Check if time is valid
             padding = max(current_time_sec * 0.05, 1.0) # At least 1s padding or 5%
             self.people_graph_plot_widget.setXRange(0, current_time_sec + padding, padding=0) # Use padding argument

        # Adjust y-axis range dynamically with padding
        if self.people_data:
            min_count = 0 # Y-axis starts at 0
            max_count_data = max(self.people_data) if self.people_data else 0
            # Consider threshold value for max range if enabled
            max_count_threshold = self.crowd_size_threshold if self.crowd_detection_enabled else 0
            max_count = max(max_count_data, max_count_threshold, 1) # Ensure range is at least 1

            y_padding = max(max_count * 0.1, 1) # 10% padding or at least 1
            # Set range from 0 up to max_count + padding
            self.people_graph_plot_widget.setYRange(min_count, max_count + y_padding, padding=0)


    def on_threshold_changed(self, value):
        """Handle confidence threshold slider change"""
        # Convert slider value (10-90) to threshold (0.1-0.9)
        self.confidence_threshold = value / 100.0

        # Update YOLO thread with new threshold
        self.yolo_thread.set_confidence_threshold(self.confidence_threshold)

    def on_heatmap_toggled(self, enabled):
        """Handle heatmap toggle switch changes"""
        self.heatmap_enabled = enabled

        # Make sure toggle visual state matches functionality
        self.heatmap_toggle.setChecked(enabled)

        # Enable/disable export button based on heatmap state AND if video is active
        can_export_heatmap = enabled and self.cap is not None and self.cap.isOpened()
        self.export_heatmap_button.setEnabled(can_export_heatmap)

        # If video is paused, reprocess the current frame
        if self.paused and self.current_frame is not None:
             # Need boxes from the last detection on this frame
             display_frame = self.process_frame_with_heatmap(self.current_frame, self.last_detected_boxes)

             if display_frame is not None:
                  # Store the updated displayed frame
                  self.displayed_frame = display_frame.copy()
                  # Convert to RGB and display
                  rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                  self.display_frame(rgb_frame)


    def on_model_changed(self, index):
        """Handle model selection change"""
        if index < 0:
            return

        # Get selected model key
        model_key = self.model_combo.itemData(index)
        if not model_key or model_key == self.current_model_key:
            return

        # Update current model
        self.current_model_key = model_key
        model_info = self.available_models[model_key]

        # Construct potential paths
        model_filename = model_info["path"]
        path_in_models_dir = os.path.join(self.models_dir, model_filename)
        path_in_current_dir = os.path.join(os.getcwd(), model_filename)


        # Check if model exists
        if os.path.exists(path_in_models_dir):
             model_path_to_use = path_in_models_dir
        elif os.path.exists(path_in_current_dir):
             model_path_to_use = path_in_current_dir
        else:
             # Model doesn't exist locally, need to download
             self.download_model(model_key)
             return # Exit, download process will handle loading later


        # Model exists locally, proceed with loading
        self.model_path = model_path_to_use
        self.yolo_ready = False

        # Update status and start loading
        self.model_status.setText(f"YOLO Model: Loading {model_key}...")
        self.model_progress.setRange(0, 0)  # Indeterminate progress
        self.model_progress.setVisible(True)
        QApplication.processEvents() # Update UI

        # Stop the existing YOLO thread if it's running
        if self.yolo_thread.isRunning():
             self.yolo_thread.stop()
             self.yolo_thread.wait() # Wait for it to finish

        # Set new model path and start YOLO thread to load the new model
        self.yolo_thread.set_model_path(self.model_path)
        self.yolo_thread.start() # This will trigger load_model inside the thread


    def download_model(self, model_key):
        """Download the selected model"""
        if self.model_downloading:
            return

        model_info = self.available_models[model_key]
        # Always save to the 'models' subdirectory
        model_save_path = os.path.join(self.models_dir, model_info["path"])

        # Update status
        self.model_status.setText(f"YOLO Model: Downloading {model_key}...")
        self.model_progress.setValue(0)
        self.model_progress.setRange(0, 100) # Set range back for percentage
        self.model_progress.setVisible(True)
        self.model_downloading = True
        QApplication.processEvents() # Update UI

        # Stop YOLO thread if it's running, as we'll need to load the new model later
        if self.yolo_thread.isRunning():
             self.yolo_thread.stop()
             self.yolo_thread.wait()

        # Create and start download thread
        self.download_thread = ModelDownloadThread(
            model_key,
            model_info["url"],
            model_save_path # Use the specific save path
        )
        self.download_thread.progress_update.connect(self.on_download_progress)
        self.download_thread.download_complete.connect(self.on_download_complete)
        self.download_thread.start()

    def on_download_progress(self, percentage, message):
        """Update download progress in UI"""
        self.model_status.setText(message)
        self.model_progress.setValue(percentage)
        QApplication.processEvents() # Ensure UI updates during download

    def on_download_complete(self, success, model_path):
        """Handle model download completion"""
        self.model_downloading = False

        if success:
            # Update model path state
            self.model_path = model_path

            # Don't automatically load model here, just update status
            self.model_status.setText(f"YOLO Model: {self.current_model_key} downloaded.")
            self.model_progress.setVisible(False)
            self.yolo_ready = False # Mark as not ready, needs loading

            # Trigger loading the newly downloaded model
            self.model_status.setText(f"YOLO Model: Loading {self.current_model_key}...")
            self.model_progress.setRange(0, 0) # Indeterminate
            self.model_progress.setVisible(True)
            QApplication.processEvents() # Update UI

            # Start YOLO thread to load the model
            if not self.yolo_thread.isRunning():
                 self.yolo_thread.set_model_path(self.model_path) # Ensure correct path
                 self.yolo_thread.start()
            # If it was already running (e.g., stopped before download), restart it
            # This case might be redundant if we always stop before download starts
            # else: # This branch might not be needed if stop() is always called before download
                 # self.yolo_thread.set_model_path(self.model_path)
                 # self.yolo_thread.start()


        else:
            # Download failed
            self.model_status.setText(f"YOLO Model: Download failed for {self.current_model_key}!")
            self.model_progress.setVisible(False)

            # Revert selection to the default model (Nano)
            default_model_key = "YOLOv8n (Nano)"
            default_model_info = self.available_models[default_model_key]
            default_model_path_local = os.path.join(self.models_dir, default_model_info["path"])
            default_model_path_cwd = os.path.join(os.getcwd(), default_model_info["path"])

            # Find the index of the default model in the combo box
            default_index = -1
            for i in range(self.model_combo.count()):
                 if self.model_combo.itemData(i) == default_model_key:
                      default_index = i
                      break

            if default_index != -1:
                 self.model_combo.setCurrentIndex(default_index) # Revert UI selection
                 # Trigger on_model_changed to attempt loading the default model
                 # This will check if the default exists or trigger its download if necessary
                 # self.on_model_changed(default_index) # Call directly
            else:
                 # Fallback if default model isn't in combo for some reason
                 self.model_status.setText("YOLO Model: Default model not found!")


    def on_model_loaded(self, success, message):
        """Handle model loading completion"""
        if success:
            self.yolo_ready = True
            # Use current_model_key which reflects the user's selection
            self.model_status.setText(f"YOLO Model: {self.current_model_key} loaded")
            self.model_progress.setVisible(False)
            # If video is paused, reprocess the current frame with the new model
            if self.paused and self.current_frame is not None:
                 self.yolo_thread.add_frame(self.current_frame) # Send frame for processing

        else:
            self.yolo_ready = False
            self.model_status.setText(f"YOLO Model: Load failed - {self.current_model_key}")
            # Optionally show the error message in tooltip or dialog
            self.model_status.setToolTip(message)
            self.model_progress.setVisible(False)
            # Consider reverting to default model if loading fails? Or just leave as failed?
            # Leaving as failed state seems reasonable.


    def open_file_dialog(self):
        """Open file dialog for selecting video files"""
        file_filter = "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        # Start in user's home directory or last used directory
        start_dir = os.path.expanduser("~")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", start_dir, file_filter
        )

        if file_path:
            self.load_video_from_path(file_path)

    def load_video_from_path(self, file_path):
        """Load and play video from the given file path"""
        print(f"Attempting to load video: {file_path}") # Debug print

        # Stop any existing video playback first
        self.stop_video() # Use the stop_video method for clean shutdown

        if not os.path.exists(file_path):
            self.video_label.set_default_content()
            self.model_status.setText("Error: Video file not found.") # Provide feedback
            print(f"Error: File not found at {file_path}")
            return

        # Make sure model is loaded or loading
        if not self.yolo_ready and not self.yolo_thread.isRunning():
             # If not ready and not even trying to load, start loading default
             print("YOLO not ready, initiating model load...")
             self.model_status.setText("YOLO Model: Loading...")
             self.model_progress.setRange(0, 0)
             self.model_progress.setVisible(True)
             QApplication.processEvents()
             self.yolo_thread.set_model_path(self.model_path) # Ensure path is set
             self.yolo_thread.start()
             # We can proceed to load the video, detection will start when model is ready

        # Initialize video capture
        try:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                print(f"Error: Failed to open video file {file_path} with OpenCV.")
                self.video_label.set_default_content()
                self.model_status.setText("Error: Could not open video file.")
                self.cap = None # Ensure cap is None if failed
                return
        except Exception as e:
             print(f"Exception opening video file {file_path}: {e}")
             self.video_label.set_default_content()
             self.model_status.setText("Error: Exception opening video.")
             self.cap = None
             return

        print("Video opened successfully.")

        # --- Reset states for new video ---
        self.video_time_ms = 0
        self.last_frame_time = time.time() # Initialize timer baseline
        self.update_timer_display()

        # Reset graph data and visual elements
        self.people_data.clear()
        self.time_data.clear()
        self.people_graph_line.setData([], [])
        if self.threshold_line: self.people_graph_plot_widget.removeItem(self.threshold_line); self.threshold_line = None
        if self.alert_segment: self.people_graph_plot_widget.removeItem(self.alert_segment); self.alert_segment = None
        if self.peak_marker: self.people_graph_plot_widget.removeItem(self.peak_marker); self.peak_marker = None
        if self.offpeak_marker: self.people_graph_plot_widget.removeItem(self.offpeak_marker); self.offpeak_marker = None

        # Reset heatmaps
        self.heatmap_accumulator = None
        self.aggregate_heatmap_accumulator = None
        self.aggregate_frame_count = 0

        # Reset counts and history
        self.people_count_history.clear()
        self.smoothed_people_count = 0
        self.people_count_value.setText("0")
        self.peak_count = 0
        self.peak_time_ms = 0
        self.offpeak_count = float('inf')
        self.offpeak_time_ms = 0
        self.peak_time_value.setText("--:--:--")
        self.peak_count_value.setText("(0 people)")
        self.offpeak_time_value.setText("--:--:--")
        self.offpeak_count_value.setText("(0 people)")

        # Reset alert state
        self.threshold_alert_active = False
        self.update_crowd_alert_status(False) # Reset visual indicator
        self.threshold_history.clear()

        # Enable controls related to video playback
        self.heatmap_toggle.setEnabled(True)
        self.crowd_toggle.setEnabled(True)
        self.export_graph_button.setEnabled(True)
        # Heatmap export enabled only if toggle is also on
        self.export_heatmap_button.setEnabled(self.heatmap_enabled)


        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.frame_interval = int(1000 / fps)  # ms between frames
        else:
            self.frame_interval = 33  # Default to ~30 fps
            print(f"Warning: Could not get FPS for {file_path}. Defaulting to {1000/self.frame_interval:.1f} FPS.")

        # Configure and start the video thread
        self.paused = False
        self.video_thread.set_capture(self.cap)
        self.video_thread.pause(False)
        if not self.video_thread.isRunning():
            self.video_thread.start()
        else:
             # If thread somehow still running (shouldn't be after stop_video),
             # ensure it uses the new capture object.
             pass # set_capture already done

        print("Video thread started.")

        # Update button states for playback
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.restart_button.setEnabled(True)
        self.end_playback_label.setVisible(False) # Hide end label


    def start_video(self):
        """Starts video playback from dropdown or resumes paused video."""
        # Case 1: Resume paused video
        if self.cap is not None and self.cap.isOpened() and self.paused:
            self.paused = False
            self.video_thread.pause(False)
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.end_playback_label.setVisible(False) # Hide end label on resume
            # Restart timer baseline on resume
            self.last_frame_time = time.time()
            return

        # Case 2: Start new video from dropdown
        # Get selected video path
        selected_index = self.source_combo.currentIndex()
        # Ensure index is valid and combobox is enabled
        if selected_index < 0 or not self.source_combo.isEnabled():
            print("No valid source selected or source combo disabled.")
            # Optionally show message to user
            self.model_status.setText("Select a valid sample source.")
            return

        video_path = self.source_combo.itemData(selected_index)

        if not video_path or not os.path.exists(video_path):
            print(f"Selected source path is invalid or does not exist: {video_path}")
            self.video_label.set_default_content()
            self.model_status.setText("Error: Selected sample video not found.")
            return

        # Use load_video_from_path to handle setup and start
        self.load_video_from_path(video_path)


    def pause_video(self):
        """Pauses or resumes video playback."""
        # Check if there is an active video capture
        if self.cap is not None and self.cap.isOpened():
            if not self.paused:
                # Pause video
                self.paused = True
                self.video_thread.pause(True)
                self.play_button.setEnabled(True)
                self.pause_button.setEnabled(False)
            else:
                # Resume video
                self.paused = False
                self.video_thread.pause(False)
                self.play_button.setEnabled(False)
                self.pause_button.setEnabled(True)
                self.end_playback_label.setVisible(False) # Hide end label on resume
                # Restart timer baseline on resume
                self.last_frame_time = time.time()


    def stop_video(self):
        """Stop video playback and reset all visualizations with improved thread handling"""
        print("Stop video requested.")
        # First, pause the video thread if it's running
        if self.video_thread is not None and self.video_thread.isRunning():
            self.paused = True # Mark as paused conceptually
            self.video_thread.pause(True)
            # Stop the video thread completely and wait for it
            self.video_thread.stop()
            self.video_thread.wait()
            print("Video thread stopped.")

        # Pause YOLO processing and clear its queue
        if hasattr(self, 'yolo_thread') and self.yolo_thread is not None:
            self.yolo_thread.frame_queue = [] # Clear pending frames
            # Don't stop the YOLO thread itself, just clear queue

        # Release video capture object safely
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
                    print("Video capture released.")
            except Exception as e:
                print(f"Error releasing video capture: {e}")
            finally:
                self.cap = None

        # Reset internal state variables
        self.paused = False # Reset pause state
        self.current_frame = None
        self.displayed_frame = None
        self.last_detected_boxes = []

        # Reset the video label to default state
        self.video_label.set_default_content()

        # Reset counts and smoothing history
        self.people_count = 0
        self.people_count_history.clear()
        self.smoothed_people_count = 0
        self.people_count_value.setText("0")
        self.people_count_value.setStyleSheet(LARGE_VALUE_FONT_STYLE) # Reset style


        # Reset video timer
        self.video_time_ms = 0
        self.last_frame_time = 0
        self.update_timer_display()
        self.end_playback_label.setVisible(False) # Hide end label

        # Reset heatmap accumulators
        self.heatmap_accumulator = None
        self.aggregate_heatmap_accumulator = None
        self.aggregate_frame_count = 0

        # Clear graph data and visual elements
        self.people_graph_plot_widget.clear() # Clears all items from the plot
        # Recreate the main plot line after clearing
        self.people_graph_line = self.people_graph_plot_widget.plot(
             [], [], pen=pg.mkPen(color=ACCENT_COLOR, width=3),
             symbolBrush=pg.mkBrush(LIGHTER_ACCENT_COLOR), symbolPen=pg.mkPen(LIGHTER_ACCENT_COLOR),
             symbolSize=4, symbol='o'
        )
        self.people_data = []
        self.time_data = []
        # Reset plot item references
        self.threshold_line = None
        self.alert_segment = None
        self.peak_marker = None
        self.offpeak_marker = None


        # Reset peak tracking display
        self.peak_count = 0
        self.peak_time_ms = 0
        self.offpeak_count = float('inf')
        self.offpeak_time_ms = 0
        self.peak_time_value.setText("--:--:--")
        self.peak_count_value.setText("(0 people)")
        self.offpeak_time_value.setText("--:--:--")
        self.offpeak_count_value.setText("(0 people)")


        # Reset threshold alert state
        self.threshold_alert_active = False
        if self.crowd_detection_enabled: # Only update visual if it was enabled
             self.update_crowd_alert_status(False)
        self.threshold_history.clear()


        # Turn off heatmap toggle and disable it if it was on
        if self.heatmap_enabled:
            self.heatmap_toggle.setChecked(False) # Update UI toggle state
            self.heatmap_enabled = False # Update internal state
        self.heatmap_toggle.setEnabled(False) # Disable toggle

        # Disable crowd detect toggle
        if self.crowd_detection_enabled:
             self.crowd_toggle.setChecked(False)
             self.crowd_detection_enabled = False
             self.crowd_settings_container.setVisible(False) # Hide settings
        self.crowd_toggle.setEnabled(False)


        # Disable export buttons
        self.export_heatmap_button.setEnabled(False)
        self.export_graph_button.setEnabled(False)

        # Reset button states to initial/stopped state
        self.play_button.setEnabled(True) # Can start a new video
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.restart_button.setEnabled(False)

        # Force UI to update
        QApplication.processEvents()
        self.repaint()
        print("Video stopped and UI reset.")


    def on_video_ended(self):
        """Handle video reaching the end"""
        print("Video ended signal received.")
        # Video thread already stops itself upon sending signal if ret is False
        # Ensure UI reflects the ended state

        self.paused = True # Treat end state as paused
        # No need to call video_thread.pause(True) if thread stopped itself

        # Show end of playback indicator
        self.end_playback_label.setVisible(True)

        # Update button states - disable play/pause, enable restart
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(True) # Still allow explicit stop
        self.restart_button.setEnabled(True)


    def update_timer_display(self):
        """Update the timer display with the current video time"""
        # Calculate hours, minutes, seconds, milliseconds
        total_seconds = self.video_time_ms // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = self.video_time_ms % 1000

        # Format the time string
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

        # Update the display
        self.timer_display.setText(time_str)

        # No need for QApplication.processEvents() here, happens frequently enough


    def process_video_frame(self, frame):
        """Process video frame received from VideoFrameThread and send to YOLO"""
        if frame is None:
            return

        # Update video timer (only if not paused)
        if not self.paused:
            current_time = time.time()
            if self.last_frame_time > 0:
                elapsed = int((current_time - self.last_frame_time) * 1000) # ms
                # Use frame interval as a more reliable timing increment if available
                # or cap elapsed time to avoid jumps during lag/resizes.
                increment = self.frame_interval if self.frame_interval > 0 else max(1, elapsed)
                # Avoid huge jumps if processing lagged significantly
                increment = min(increment, 250) # Cap increment to e.g., 250ms
                self.video_time_ms += increment
            self.last_frame_time = current_time
            self.update_timer_display()

        # Store the raw frame (for heatmap processing later maybe, or resize)
        self.current_frame = frame.copy()

        # Send the frame to YOLO thread for detection if model is ready
        if self.yolo_ready and self.yolo_thread.model:
            self.yolo_thread.add_frame(frame)
        else:
            # If YOLO is not ready, display the raw frame without detection/heatmap
            # Process with empty boxes list for consistency
            display_frame_no_yolo = self.process_frame_with_heatmap(frame, [])

            if display_frame_no_yolo is not None:
                 self.displayed_frame = display_frame_no_yolo.copy()
                 rgb_frame = cv2.cvtColor(self.displayed_frame, cv2.COLOR_BGR2RGB)
                 self.display_frame(rgb_frame)
            # If YOLO is loading, status message should already indicate that


    def update_peak_time_display(self):
        """Update peak and off-peak time displays and markers"""
        # Peak time update
        if self.peak_time_ms >= 0 and self.peak_count > 0: # Check count > 0
            peak_hours = (self.peak_time_ms // 1000) // 3600
            peak_minutes = ((self.peak_time_ms // 1000) % 3600) // 60
            peak_seconds = (self.peak_time_ms // 1000) % 60
            peak_time_str = f"{peak_hours:02d}:{peak_minutes:02d}:{peak_seconds:02d}"
            self.peak_time_value.setText(peak_time_str)
            self.peak_count_value.setText(f"({self.peak_count} people)")

            # Update peak marker on graph
            if self.people_data: # Check if graph data exists
                 peak_time_sec = self.peak_time_ms / 1000.0
                 if self.peak_marker is None:
                      self.peak_marker = self.people_graph_plot_widget.plot(
                           [peak_time_sec], [self.peak_count],
                           pen=None, symbol='o', symbolSize=10,
                           symbolBrush='#FF5555' # Red
                      )
                 else:
                      self.peak_marker.setData([peak_time_sec], [self.peak_count])

        # Off-peak time update
        if self.offpeak_time_ms >= 0 and self.offpeak_count < float('inf'): # Check if valid count found
            offpeak_hours = (self.offpeak_time_ms // 1000) // 3600
            offpeak_minutes = ((self.offpeak_time_ms // 1000) % 3600) // 60
            offpeak_seconds = (self.offpeak_time_ms // 1000) % 60
            offpeak_time_str = f"{offpeak_hours:02d}:{offpeak_minutes:02d}:{offpeak_seconds:02d}"
            self.offpeak_time_value.setText(offpeak_time_str)
            self.offpeak_count_value.setText(f"({self.offpeak_count} people)")

            # Update off-peak marker on graph
            if self.people_data: # Check if graph data exists
                 offpeak_time_sec = self.offpeak_time_ms / 1000.0
                 if self.offpeak_marker is None:
                      self.offpeak_marker = self.people_graph_plot_widget.plot(
                           [offpeak_time_sec], [self.offpeak_count],
                           pen=None, symbol='o', symbolSize=10,
                           symbolBrush='#5599FF' # Blue
                      )
                 else:
                      self.offpeak_marker.setData([offpeak_time_sec], [self.offpeak_count])


    def display_detection_results(self, processed_frame_with_boxes, people_count, boxes):
        """Display frame processed by YOLO, update counts, graph, heatmap, etc."""
        # Note: processed_frame_with_boxes already has boxes drawn by YOLO thread
        if processed_frame_with_boxes is None:
            return

        # Store the last detected boxes for use when toggling heatmap while paused
        self.last_detected_boxes = boxes.copy()

        # Add current raw count to history for smoothing
        self.people_count_history.append(people_count)

        # Calculate smoothed people count (moving average)
        if len(self.people_count_history) > 0:
            new_smoothed_count = round(np.mean(self.people_count_history))
        else:
            new_smoothed_count = people_count # Should not happen if history is appended first

        # Update internal state only if smoothed count changes
        # This prevents unnecessary UI updates if count is stable
        smoothed_count_changed = (new_smoothed_count != self.smoothed_people_count)
        self.smoothed_people_count = new_smoothed_count

        # Update people count display
        # Always update display text even if value is same, in case style needs resetting (e.g. after alert)
        self.people_count_value.setText(str(self.smoothed_people_count))
        # Reset style if not in alert state
        if not self.threshold_alert_active:
             self.people_count_value.setStyleSheet(LARGE_VALUE_FONT_STYLE)


        # Check for threshold crossing if crowd detection is enabled
        if self.crowd_detection_enabled:
            self.check_threshold_crossing(processed_frame_with_boxes) # Pass frame for potential visualization


        # Update the people count graph with smoothed value
        self.update_people_graph(self.smoothed_people_count)

        # Track peak and off-peak based on smoothed count
        if self.smoothed_people_count > self.peak_count:
            self.peak_count = self.smoothed_people_count
            self.peak_time_ms = self.video_time_ms
            self.update_peak_time_display()

        # Track off-peak only if count is positive and lower than current off-peak
        if self.smoothed_people_count > 0 and self.smoothed_people_count < self.offpeak_count:
            self.offpeak_count = self.smoothed_people_count
            self.offpeak_time_ms = self.video_time_ms
            self.update_peak_time_display()


        # --- Frame Display ---
        # The frame received ('processed_frame_with_boxes') already has YOLO boxes drawn.
        # We now apply heatmap (if enabled) and alert borders (if active) on top of this.
        # We use the raw self.current_frame for heatmap calculations, but overlay onto the processed frame.

        # Get the frame to display by applying heatmap/alerts onto the YOLO output frame
        final_display_frame = self.process_frame_with_heatmap(processed_frame_with_boxes, boxes)

        if final_display_frame is not None:
             # Store the final frame that includes heatmap/alerts
             self.displayed_frame = final_display_frame.copy()

             # Convert to RGB for display
             rgb_frame = cv2.cvtColor(final_display_frame, cv2.COLOR_BGR2RGB)

             # Display the processed frame
             self.display_frame(rgb_frame)


    def check_threshold_crossing(self, frame):
        """Check if smoothed people count exceeds threshold and update alert status."""
        # Determine if alert should be active based on smoothed count vs threshold
        should_alert_be_active = (self.smoothed_people_count > self.crowd_size_threshold)

        # Update the visual status only if the state changes
        if should_alert_be_active != self.threshold_alert_active:
             self.update_crowd_alert_status(should_alert_be_active, self.smoothed_people_count)
             # No need to redraw frame here, display_detection_results handles it


    def display_frame(self, rgb_frame):
        """Display a video frame in the video_label, scaling it correctly."""
        if rgb_frame is None or not hasattr(self, 'video_label') or not self.video_label.isVisible():
             # print("Debug: Skipping display_frame (no frame or label not ready)")
             return # Don't process if no frame or label isn't ready

        try:
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Get the current size of the video label
            label_size = self.video_label.size()
            if label_size.isEmpty() or label_size.width() <= 0 or label_size.height() <= 0:
                 # print("Debug: Skipping display_frame (label size invalid)")
                 # Label might not be fully initialized yet
                 return

            # Scale the pixmap to fit the label size while preserving aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(label_size,
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)

            # Hide the default content (icon/text) widgets before showing video
            # This check might only be needed once, could be optimized
            if self.video_label.icon_label.isVisible():
                 self.video_label.icon_label.setVisible(False)
                 self.video_label.text_label.setVisible(False)


            # Display the frame
            self.video_label.setPixmap(scaled_pixmap)
            # Ensure alignment remains center even after setting pixmap
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        except Exception as e:
             print(f"Error in display_frame: {e}")
             # Optionally, reset to default content on error?
             # self.video_label.set_default_content()


    def resizeEvent(self, event):
        """Handle window resize events, redraw current frame."""
        super().resizeEvent(event)

        # Redraw the currently displayed frame (could be raw or processed)
        # Use self.displayed_frame as it represents what *should* be shown
        if self.displayed_frame is not None:
            try:
                 rgb_frame = cv2.cvtColor(self.displayed_frame, cv2.COLOR_BGR2RGB)
                 self.display_frame(rgb_frame)
            except cv2.error as e:
                 print(f"CV2 Error during resize redraw: {e}")
            except Exception as e:
                 print(f"Error during resize redraw: {e}")
        elif self.cap is None: # If no video loaded, ensure default content is shown
             self.video_label.set_default_content()


    def closeEvent(self, event):
        """Handle application close event cleanly."""
        print("Close event triggered.")
        # Stop the video thread first
        if self.video_thread is not None and self.video_thread.isRunning():
            print("Stopping video thread...")
            self.video_thread.stop()
            self.video_thread.wait() # Wait for clean exit
            print("Video thread stopped.")

        # Stop the YOLO thread
        if self.yolo_thread is not None and self.yolo_thread.isRunning():
            print("Stopping YOLO thread...")
            self.yolo_thread.stop()
            self.yolo_thread.wait() # Wait for clean exit
            print("YOLO thread stopped.")

        # Stop download thread if running
        if self.download_thread is not None and self.download_thread.isRunning():
             print("Stopping download thread...")
             # Download thread might not have a stop method, terminate might be needed
             # Or rely on application exit to kill it. Let's assume wait is sufficient.
             # self.download_thread.stop() # If it had one
             self.download_thread.wait()
             print("Download thread finished.")


        # Release video capture
        if self.cap is not None and self.cap.isOpened():
            print("Releasing video capture...")
            self.cap.release()
            self.cap = None
            print("Video capture released.")

        print("Accepting close event.")
        event.accept()


    def export_count_graph(self):
        """Export the people count graph as an image"""
        # Import matplotlib here to avoid loading it unless exporting
        try:
             import matplotlib.pyplot as plt
             from matplotlib.figure import Figure
             from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        except ImportError:
             self.show_export_error_message("Matplotlib is required for graph export. Please install it (`pip install matplotlib`).")
             return

        # Check if we have graph data
        if not self.time_data or not self.people_data:
            self.show_export_error_message("No graph data available to export.")
            return

        # Ask the user to select an output file/location
        default_filename = f"people_count_graph_{time.strftime('%Y%m%d-%H%M%S')}.png"
        default_path = os.path.join(os.getcwd(), "exports", default_filename)
        # Ensure exports directory exists for suggestion
        os.makedirs(os.path.join(os.getcwd(), "exports"), exist_ok=True)

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save People Count Graph", default_path, "PNG Images (*.png);;All Files (*)"
        )

        if not output_path:  # User canceled
            return

        # Ensure filename ends with .png
        if not output_path.lower().endswith(".png"):
             output_path += ".png"

        # Ensure the chosen directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
             try:
                  os.makedirs(output_dir, exist_ok=True)
             except OSError as e:
                  self.show_export_error_message(f"Could not create output directory:\n{output_dir}\nError: {e}")
                  return


        # Create a high-resolution figure
        try:
             fig = Figure(figsize=(12, 7), dpi=150) # Slightly larger figure
             canvas = FigureCanvas(fig)
             ax = fig.add_subplot(111)

             # Plot the data with styling
             ax.plot(list(self.time_data), list(self.people_data),
                     marker='o', markersize=4, linewidth=2, color=ACCENT_COLOR, label='People Count')

             # Add threshold line if it was enabled
             if self.crowd_detection_enabled:
                  ax.axhline(y=self.crowd_size_threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({self.crowd_size_threshold})')

             # Add peak/off-peak markers if they exist
             if self.peak_marker is not None and self.peak_count > 0:
                  ax.plot(self.peak_time_ms / 1000.0, self.peak_count, 'o', markersize=8, color='#FF5555', label=f'Peak ({self.peak_count})')
             if self.offpeak_marker is not None and self.offpeak_count < float('inf'):
                   ax.plot(self.offpeak_time_ms / 1000.0, self.offpeak_count, 'o', markersize=8, color='#5599FF', label=f'Off-Peak ({self.offpeak_count})')


             # Style the plot to match UI theme
             ax.set_facecolor(WIDGET_BG_COLOR)
             fig.patch.set_facecolor(PANEL_BG_COLOR)

             # Grid styling
             ax.grid(True, linestyle='--', alpha=0.3, color='#888888')

             # Spine styling (borders)
             for spine in ax.spines.values():
                  spine.set_color(BORDER_COLOR)

             # Set labels and title
             ax.set_xlabel('Time (seconds)', color=TEXT_COLOR)
             ax.set_ylabel('People Count', color=TEXT_COLOR)
             ax.set_title('People Count Over Time', color=TEXT_COLOR, fontsize=14, weight='bold') # Match UI text

             # Style the ticks
             ax.tick_params(axis='x', colors=MUTED_TEXT_COLOR)
             ax.tick_params(axis='y', colors=MUTED_TEXT_COLOR)

             # Add legend if threshold or markers were added
             if self.crowd_detection_enabled or self.peak_marker or self.offpeak_marker:
                  legend = ax.legend()
                  plt.setp(legend.get_texts(), color=MUTED_TEXT_COLOR) # Style legend text
                  legend.get_frame().set_facecolor(WIDGET_BG_COLOR)
                  legend.get_frame().set_edgecolor(BORDER_COLOR)


             # Adjust layout
             fig.tight_layout()

             # Save the figure
             fig.savefig(output_path)
             print(f"Graph saved to: {output_path}")

             # Show success message
             self.show_export_success_message(output_path)

        except Exception as e:
             print(f"Error during graph export: {e}")
             self.show_export_error_message(f"Failed to export graph:\n{e}")
        finally:
             # Ensure matplotlib figure is closed to release memory
             if 'fig' in locals():
                  plt.close(fig)



    def export_heatmap(self):
        """Export the aggregate heatmap directly after selecting a file location"""
        # Check if we have aggregate heatmap data
        if self.aggregate_heatmap_accumulator is None or self.aggregate_frame_count <= 0:
            self.show_export_error_message("No heatmap data collected yet. Play a video with heatmap enabled first.")
            return

        # Also check if we have a reference frame dimensions
        if self.current_frame is None and self.displayed_frame is None:
             self.show_export_error_message("Cannot determine heatmap size. Play or load a video first.")
             return

        # Determine output size from current/displayed frame
        ref_frame = self.current_frame if self.current_frame is not None else self.displayed_frame
        h, w = ref_frame.shape[:2]

        # Ask the user to select an output file/location
        default_filename = f"aggregate_heatmap_{time.strftime('%Y%m%d-%H%M%S')}.png"
        default_path = os.path.join(os.getcwd(), "exports", default_filename)
        # Ensure exports directory exists for suggestion
        os.makedirs(os.path.join(os.getcwd(), "exports"), exist_ok=True)

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Aggregate Heatmap", default_path, "PNG Images (*.png);;All Files (*)"
        )

        if not output_path:  # User canceled
            return

        # Ensure filename ends with .png
        if not output_path.lower().endswith(".png"):
             output_path += ".png"

        # Ensure the chosen directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
             try:
                  os.makedirs(output_dir, exist_ok=True)
             except OSError as e:
                  self.show_export_error_message(f"Could not create output directory:\n{output_dir}\nError: {e}")
                  return

        try:
             # Create a normalized version of the aggregate heatmap
             # Normalize by the max value in the aggregate heatmap for better contrast
             aggregate_norm = self.aggregate_heatmap_accumulator.copy()
             max_aggr_val = np.max(aggregate_norm)

             if max_aggr_val > 0:
                  aggregate_norm /= max_aggr_val
             else:
                  print("Warning: Aggregate heatmap has no intensity.")
                  # Result will be black or just the background frame

             # Upsample normalized aggregate heatmap to original frame size
             heatmap_resized = cv2.resize(aggregate_norm, (w, h), interpolation=cv2.INTER_LINEAR)

             # Apply additional blur for smoother visualization (optional, but can look better)
             heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (21, 21), 0) # Kernel size adjustable

             # Convert blurred heatmap to colormap
             # Ensure values are clipped between 0 and 1 before scaling to 255
             heatmap_clipped = np.clip(heatmap_blurred, 0, 1)
             heatmap_8bit = (heatmap_clipped * 255).astype(np.uint8)
             heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

             # Create a background using the last displayed frame if available, else black
             if self.displayed_frame is not None:
                 background = cv2.addWeighted(self.displayed_frame, 0.4, np.zeros_like(self.displayed_frame), 0.6, 0)
                 result = cv2.addWeighted(heatmap_colored, 0.7, background, 0.3, 0) # Blend heatmap over background
             else:
                 # Fallback to just the colored heatmap if no background frame
                 result = heatmap_colored

             # Save the result
             cv2.imwrite(output_path, result)
             print(f"Heatmap saved to: {output_path}")

             # Show success message
             self.show_export_success_message(output_path)

        except Exception as e:
             print(f"Error exporting heatmap: {e}")
             self.show_export_error_message(f"Failed to export heatmap:\n{e}")


    def show_export_success_message(self, output_path):
        """Show success message for heatmap/graph export"""
        # Import platform-specific tools here
        import subprocess
        import platform

        # Check if the path actually exists before showing message
        if not os.path.exists(output_path):
             print(f"Export path does not exist after saving: {output_path}")
             self.show_export_error_message(f"File not found after saving:\n{output_path}")
             return

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Export Complete")
        msg.setText("Export completed successfully!")
        msg.setInformativeText(f"File saved to:\n{output_path}")
        # Offer to open the file's location
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Open)
        open_button = msg.button(QMessageBox.StandardButton.Open)
        open_button.setText("Open Location") # More accurate text

        result = msg.exec()

        if result == QMessageBox.StandardButton.Open:
            try:
                 output_dir = os.path.dirname(output_path)
                 if platform.system() == "Windows":
                      os.startfile(output_dir)
                 elif platform.system() == "Darwin":  # macOS
                      subprocess.Popen(["open", output_dir])
                 else:  # Linux and other Unix-like
                      subprocess.Popen(["xdg-open", output_dir])
            except Exception as e:
                 print(f"Error opening directory {output_dir}: {e}")
                 # Show a secondary message if opening fails
                 QMessageBox.warning(self, "Open Location Failed", f"Could not open the directory:\n{output_dir}\nError: {e}")


    def show_export_error_message(self, error_msg):
        """Show error message for export failures"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Export Error")
        msg.setText("Could not complete export.")
        msg.setInformativeText(error_msg)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()