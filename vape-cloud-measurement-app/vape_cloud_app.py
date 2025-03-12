#!/usr/bin/env python3
"""
Vape Cloud Measurement App
---------------------------
A desktop application that uses the webcam feed to detect vape clouds in real-time,
highlight them with a bounding box, and measure the (relative) size of the detected cloud.
The GUI is developed using PyQt5, while OpenCV handles video capture and image processing.

Core functionalities include:
   - Webcam selection and live feed display.
   - Real-time vape cloud detection and measurement.
   - Adjustable detection sensitivity via a settings slider.
   - Data logging (optional – if recording is enabled, measurements are saved for later export).
   - A modern, aesthetically pleasing dark-style GUI.

Run this file to start the application.
"""

import sys
import cv2
import time
import csv
import numpy as np
from datetime import datetime

# PyQt5 imports:
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QSlider, QCheckBox, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont

# ------------------------------------------------------------------------------------
# Video Capture Thread
# ------------------------------------------------------------------------------------
class VideoThread(QThread):
    # Signal to send the processed video frame (as an image array)
    change_pixmap_signal = pyqtSignal(np.ndarray)
    # Signal to send the current measurement (in pixel units)
    measurement_signal = pyqtSignal(float)

    def __init__(self, cam_index=0, sensitivity=50):
        super().__init__()
        self._run_flag = True
        self.camera_index = cam_index
        self.sensitivity = sensitivity  # For vape detection (adjustable)
    
    def run(self):
        # Open the selected webcam
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Error: Could not open video device.")
            self._run_flag = False

        # Process frames until the thread is stopped
        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Process the frame using our vape cloud detection function:
                processed_frame, measurement = detect_vape_cloud(frame, self.sensitivity)
                # Emit the signals with the processed frame and numeric measurement
                self.change_pixmap_signal.emit(processed_frame)
                self.measurement_signal.emit(measurement)
            else:
                print("Failed to grab frame")
            # Tiny sleep to reduce CPU load:
            cv2.waitKey(1)
        # Release the camera when done
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def update_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity

    def update_camera_index(self, cam_index):
        self.camera_index = cam_index

# ------------------------------------------------------------------------------------
# Vape Cloud Detection Function
# ------------------------------------------------------------------------------------
def detect_vape_cloud(frame, sensitivity=50):
    """
    Process the frame to detect a vape cloud.
    
    Idea:
      - Convert the frame to HSV color space.
      - Create a mask to detect “cloud-like” regions (here we look for light regions).
      - Use morphological filtering to reduce noise.
      - Find contours and pick the largest.
      - Draw a bounding box and return the area of that box.
    
    Parameters:
        frame (np.ndarray): The input image frame (BGR format).
        sensitivity (int): A value (0-255) controlling mask thresholding.
    
    Returns:
        processed_frame (np.ndarray): The frame with an overlay (if cloud is detected).
        measurement (float): The area (in pixel units) of the detected vape cloud.
    """
    # Convert to HSV – vape clouds are usually whitish so we threshold on high brightness.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define HSV thresholds for “white” vapor (the range can be tuned)
    lower_white = np.array([0, 0, sensitivity])
    upper_white = np.array([180, 55, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply morphological opening (erosion followed by dilation) to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask. Use RETR_EXTERNAL to get only the outer contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measurement = 0  # Default measurement if nothing is found

    if contours:
        # Choose the largest contour assuming it is the vape cloud.
        largest_contour = max(contours, key=cv2.contourArea)
        # Get bounding box around the contour.
        x, y, w, h = cv2.boundingRect(largest_contour)
        measurement = w * h  # Relative area in pixels.
        # Draw a rectangle on the original frame (in green)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Optionally, overlay the measurement as text:
        cv2.putText(frame, f"Size: {measurement:,} px", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame, measurement

# ------------------------------------------------------------------------------------
# Main Application Window
# ------------------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vape Cloud Measurement App")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2e2e2e; color: #FFF;")
        
        # Container widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        
        # Webcam preview label (for video frames)
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000;")
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        
        # Measurement readout label
        self.measurement_label = QLabel("Cloud Size: 0 px")
        self.measurement_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.measurement_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.measurement_label)
        
        # Controls layout: Start/Stop, Camera Selection, Sensitivity, and Data Logging
        controls_layout = QHBoxLayout()
        
        # Start/Stop button
        self.start_button = QPushButton("Start Measurement")
        self.start_button.clicked.connect(self.start_video)
        controls_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_video)
        controls_layout.addWidget(self.stop_button)
        
        # Camera selection dropdown – try testing indices 0–3
        self.cam_selector = QComboBox()
        self.populate_camera_list()
        controls_layout.addWidget(self.cam_selector)
        
        # Sensitivity slider (range 0 to 255, default 50); higher sensitivity means lighter threshold.
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(0)
        self.sensitivity_slider.setMaximum(255)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setToolTip("Adjust detection sensitivity")
        self.sensitivity_slider.valueChanged.connect(self.sensitivity_changed)
        controls_layout.addWidget(QLabel("Sensitivity:"))
        controls_layout.addWidget(self.sensitivity_slider)

        # Checkbox for data logging (if checked, measurements are stored)
        self.log_checkbox = QCheckBox("Record Data")
        controls_layout.addWidget(self.log_checkbox)

        # Button to export the recorded data to CSV
        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self.export_csv)
        controls_layout.addWidget(self.export_button)
        
        main_layout.addLayout(controls_layout)
        
        # List to store measurement data as tuples: (timestamp, measurement)
        self.measurement_data = []
        
        # Video thread (initially not running)
        self.thread = None

    def populate_camera_list(self):
        """
        In many setups the default webcam is at index 0. Here we simply attempt indices 0-3.
        In a production product, you might scan system devices.
        """
        self.cam_selector.clear()
        available_indices = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_indices.append(i)
                cap.release()
        # If no camera is found, add index 0 to avoid error.
        if not available_indices:
            available_indices = [0]
        for idx in available_indices:
            self.cam_selector.addItem(f"Camera {idx}", idx)

    def start_video(self):
        """Start the video capture thread and update UI elements."""
        cam_index = self.cam_selector.currentData()
        sensitivity = self.sensitivity_slider.value()
        # Create and start the video thread:
        self.thread = VideoThread(cam_index=cam_index, sensitivity=sensitivity)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.measurement_signal.connect(self.update_measurement)
        self.thread.start()
        # Update buttons and disable camera selector to avoid changes while running:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cam_selector.setEnabled(False)
        # Clear previous measurement data if logging is enabled:
        if self.log_checkbox.isChecked():
            self.measurement_data = []

    def stop_video(self):
        """Stop the video capture thread and update UI elements."""
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cam_selector.setEnabled(True)

    def update_image(self, cv_img):
        """
        Convert the image from OpenCV BGR format to Qt's QImage and display it.
        """
        # Convert image to RGB:
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # Scale the QImage to the video_label dimensions:
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    def update_measurement(self, value):
        """Update the measurement label; log data if enabled."""
        self.measurement_label.setText(f"Cloud Size: {value:,} px")
        if self.log_checkbox.isChecked():
            timestamp = datetime.now().isoformat(timespec="seconds")
            self.measurement_data.append((timestamp, value))

    def sensitivity_changed(self, value):
        """Update sensitivity in the video thread if running."""
        if self.thread is not None:
            self.thread.update_sensitivity(value)

    def export_csv(self):
        """Export logged data to a CSV file if any data exists."""
        if not self.measurement_data:
            print("No measurement data to export.")
            return
        # Ask user to choose save location:
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Measurement Data", "", "CSV Files (*.csv)", options=options
        )
        if filename:
            try:
                with open(filename, "w", newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(["Timestamp", "Cloud Size (px)"])
                    csvwriter.writerows(self.measurement_data)
                print(f"Data exported successfully to {filename}")
            except Exception as e:
                print("Failed to export data:", e)

    def closeEvent(self, event):
        """Ensure that the video thread stops when the application is closed."""
        if self.thread is not None:
            self.thread.stop()
        event.accept()

# ------------------------------------------------------------------------------------
# Application Entry Point
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())