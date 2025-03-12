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