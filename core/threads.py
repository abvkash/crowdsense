import sys
import os
import cv2
import torch
import numpy as np
import urllib.request
import time

from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

class ModelDownloadThread(QThread):
    """Thread for downloading YOLO models"""
    progress_update = pyqtSignal(int, str)  # Progress percentage, message
    download_complete = pyqtSignal(bool, str)  # Success, model path

    def __init__(self, model_name, model_url, save_path):
        super().__init__()
        self.model_name = model_name
        self.model_url = model_url
        self.save_path = save_path

    def run(self):
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.download_with_progress(self.model_url, self.save_path)
            self.progress_update.emit(100, f"Model {self.model_name} downloaded successfully")
            self.download_complete.emit(True, self.save_path)

        except Exception as e:
            error_msg = f"Error downloading model {self.model_name}: {str(e)}"
            print(error_msg)
            self.progress_update.emit(0, error_msg)
            self.download_complete.emit(False, "")

    def download_with_progress(self, url, save_path):
        """Download a file with progress reporting"""

        def progress_callback(count, block_size, total_size):
            if total_size > 0:
                percentage = min(int(count * block_size * 100 / total_size), 100)
                self.progress_update.emit(percentage, f"Downloading {self.model_name}: {percentage}%")

        self.progress_update.emit(0, f"Starting download of {self.model_name}...")
        urllib.request.urlretrieve(url, save_path, progress_callback)

class VideoFrameThread(QThread):
    """Separate thread for handling video frames to prevent UI slowdowns"""
    frame_ready = pyqtSignal(object)
    video_ended = pyqtSignal()  # Signal when video reaches end - at class level

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.paused = False
        self.loop_detected = False

    def set_capture(self, cap):
        self.cap = cap

    def stop(self):
        self.running = False
        self.wait()

    def pause(self, paused):
        self.paused = paused

    def run(self):
        self.running = True

        # For local videos or webcams
        while self.running and self.cap is not None and self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    # Video ended - don't automatically restart
                    # Just emit the end-of-video signal
                    self.video_ended.emit()
                    # Important: Stop the loop after emitting video_ended if ret is False
                    # Otherwise it keeps trying to read from a finished source
                    self.running = False

            # Sleep to control frame rate
            self.msleep(30)  # ~33 fps

class YoloDetectionThread(QThread):
    """Separate thread for YOLO detection to prevent UI slowdowns"""
    detection_ready = pyqtSignal(object, int, list)  # Frame, count, boxes
    model_loaded = pyqtSignal(bool, str)  # Success, message

    def __init__(self, model_path="yolov8n.pt"):
        super().__init__()
        self.frame_queue = []
        self.running = False
        self.model = None
        self.model_path = model_path
        self.processing = False
        self.loading_model = False
        self.confidence_threshold = 0.4  # Default threshold

    def set_model_path(self, model_path):
        """Set a new model path and reset the model"""
        self.model_path = model_path
        self.model = None

    def add_frame(self, frame):
        if frame is not None and not self.processing:
            self.frame_queue = [frame.copy()]  # Only keep the latest frame

    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for detections"""
        self.confidence_threshold = threshold

    def stop(self):
        self.running = False
        self.wait()

    def load_model(self):
        """Load YOLO model"""
        if self.loading_model:
            return

        self.loading_model = True

        try:
            self.model = YOLO(self.model_path)
            self.model_loaded.emit(True, f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            error_msg = f"Error loading YOLO model: {e}"
            self.model_loaded.emit(False, error_msg)

        self.loading_model = False

    def run(self):
        self.running = True

        # Load YOLO model if not already loaded
        if self.model is None:
            self.load_model()

        while self.running:
            if len(self.frame_queue) > 0 and self.model is not None:
                self.processing = True
                frame = self.frame_queue.pop(0)

                try:
                    # Run YOLO detection on the frame
                    results = self.model(frame, classes=0)  # Class 0 is 'person' in COCO dataset

                    people_count = 0
                    boxes = []

                    for result in results:
                        result_boxes = result.boxes
                        for box in result_boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            confidence = float(box.conf[0])

                            if confidence > self.confidence_threshold:
                                # Draw bounding box
                                color = (0, 255, 0)  # Green
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                                conf_text = f"{confidence:.2f}"
                                cv2.putText(frame, conf_text, (x1, y1-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                                # Store box coordinates for heatmap
                                boxes.append((x1, y1, x2, y2))

                                people_count += 1

                    # Emit the processed frame, people count, and boxes for heatmap
                    self.detection_ready.emit(frame, people_count, boxes)

                except Exception as e:
                    print(f"Error in YOLO detection: {e}")

                self.processing = False

            self.msleep(10)
