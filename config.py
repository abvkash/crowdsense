# Constants for styling
DARK_BG_COLOR = "#1E1E1E"
PANEL_BG_COLOR = "#252526"
WIDGET_BG_COLOR = "#2D2D30"
DROPDOWN_BG_COLOR = "#333337"
BORDER_COLOR = "#3E3E42"
TEXT_COLOR = "#E0E0E0"
MUTED_TEXT_COLOR = "#AAAAAA"
ACCENT_COLOR = "#007ACC"
LIGHTER_ACCENT_COLOR = "#1C97EA"
DARKER_ACCENT_COLOR = "#005A9C"
GRID_COLOR = (80, 80, 80)  # For OpenCV which uses RGB tuples

# Font styles
DEFAULT_FONT = "font-family: Arial;"
HEADER_FONT_STYLE = f"{DEFAULT_FONT} font-size: 16px; font-weight: bold; color: {TEXT_COLOR}; border: none;"
SUBHEADER_FONT_STYLE = f"{DEFAULT_FONT} font-size: 14px; color: {TEXT_COLOR}; border: none;"
VALUE_FONT_STYLE = f"{DEFAULT_FONT} font-size: 14px; font-weight: bold; color: {ACCENT_COLOR}; border: none;"
LARGE_VALUE_FONT_STYLE = f"{DEFAULT_FONT} font-size: 32px; font-weight: bold; color: {ACCENT_COLOR}; border: none;"

# Button style templates
BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {DARKER_ACCENT_COLOR};
        color: white;
        border: 1px solid {ACCENT_COLOR};
        border-radius: 3px;
        {DEFAULT_FONT}
        font-size: 13px;
        padding: 0px;
        min-width: 32px;
        min-height: 32px;
        max-width: 32px;
        max-height: 32px;
        margin: 0px;
        text-align: center;
    }}

    QPushButton:hover {{
        background-color: {ACCENT_COLOR};
        border: 1px solid {LIGHTER_ACCENT_COLOR};
    }}

    QPushButton:pressed {{
        background-color: {DARKER_ACCENT_COLOR};
        border: 1px solid {ACCENT_COLOR};
    }}

    QPushButton:checked {{
        background-color: {DARKER_ACCENT_COLOR};
        border: 1px solid {ACCENT_COLOR};
    }}

    QPushButton:disabled {{
        background-color: #3E3E3E;
        color: #888888;
        border: 1px solid #505050;
    }}
"""

EXPORT_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {DARKER_ACCENT_COLOR};
        color: white;
        border: 1px solid {ACCENT_COLOR};
        border-radius: 3px;
        {DEFAULT_FONT}
        font-size: 13px;
        padding: 6px 12px;
    }}

    QPushButton:hover {{
        background-color: {ACCENT_COLOR};
        border: 1px solid {LIGHTER_ACCENT_COLOR};
    }}

    QPushButton:pressed {{
        background-color: {DARKER_ACCENT_COLOR};
        border: 1px solid {ACCENT_COLOR};
    }}

    QPushButton:disabled {{
        background-color: #3E3E3E;
        color: #888888;
        border: 1px solid #505050;
    }}
"""

# Define available YOLO models
available_models = {
    "YOLOv8n (Nano)": {
        "path": "yolov8n.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "description": "Smallest and fastest model, best for weaker hardware",
        "size": "6.2 MB"
    },
    "YOLOv8s (Small)": {
        "path": "yolov8s.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "description": "Good balance of speed and accuracy",
        "size": "21.5 MB"
    },
    "YOLOv8m (Medium)": {
        "path": "yolov8m.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "description": "Better accuracy, still reasonable performance",
        "size": "51.5 MB"
    },
    "YOLOv8l (Large)": {
        "path": "yolov8l.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "description": "High accuracy, slower performance",
        "size": "87.5 MB"
    },
    "YOLOv8x (XLarge)": {
        "path": "yolov8x.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
        "description": "Best accuracy, slowest performance",
        "size": "136.5 MB"
    }
}
