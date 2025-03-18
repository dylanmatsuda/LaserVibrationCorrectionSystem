import sys
import threading
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from simple_pid import PID
import nidaqmx
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_GAIN, -6)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Thread-safe frame storage
frame_lock = threading.Lock()
latest_frame = None
running = True
processing = False  # Flag to track if processing is active

frame_counter = 0

# Storage for Centroid Data
centroid_x_tot = []
centroid_y_tot = []
time_list = []

# Define Target Position
TARGET_X = 305
TARGET_Y = 235

# Step response test variables
STEP_DELAY = 5
STEP_AMOUNT = -50

# DAQ setup
# daq_task_x = nidaqmx.Task()
# daq_task_y = nidaqmx.Task()
# daq_task_x.ao_channels.add_ao_voltage_chan("Dev1/ao0")
# daq_task_y.ao_channels.add_ao_voltage_chan("Dev1/ao1")

# PID Controllers
pid_x = PID(0.0045, 0, 0.000, setpoint=0)
pid_y = PID(0, 0, 0, setpoint=TARGET_Y)
pid_x.output_limits = (-10, 10)
pid_y.output_limits = (-10, 10)

voltage_x_tot = 0
voltage_y_tot = 0

def send_daq_signal(error_x, error_y):
    """ Send correction signal using PID control """
    global voltage_x_tot
    voltage_x_tot += pid_x(error_x)
    voltage_y = pid_y(error_y)
    # daq_task_x.write(voltage_x_tot)
    # daq_task_y.write(voltage_y)

def capture_frames():
    """ Continuously captures frames for GUI display. """
    global latest_frame, running, frame_counter
    while running:
        ret, frame = cap.read()
        if not ret:
            # time.sleep(0.1)
            continue
        with frame_lock:
            latest_frame = frame.copy()

# def process_frame(window):
#     """ Tracks and corrects laser position using PID. Automatically plots data when finished. """
#     global processing, TARGET_X, TARGET_Y
#     processing = True
#     start_time = time.time()
#     step_applied = False
#
#     centroid_x_tot.clear()
#     centroid_y_tot.clear()
#     time_list.clear()
#
#     while processing:
#         with frame_lock:
#             if latest_frame is None:
#                 continue
#             frame = latest_frame.copy()
#
#         red_channel = frame[:, :, 2]
#         blurred = cv2.GaussianBlur(red_channel, (5, 5), 0)
#         _, red_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         cx, cy = None, None
#         if len(contours) > 0:
#             contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
#             centroids = [(int(cv2.moments(cnt)["m10"] / cv2.moments(cnt)["m00"]),
#                           int(cv2.moments(cnt)["m01"] / cv2.moments(cnt)["m00"]),
#                           cv2.moments(cnt)["m00"]) for cnt in contours if cv2.moments(cnt)["m00"] > 0]
#
#             if centroids:
#                 cx, cy, _ = max(centroids, key=lambda c: c[2])  # Largest centroid
#
#         elapsed_time = time.time() - start_time
#         if elapsed_time > STEP_DELAY and not step_applied:
#             step_applied = True
#             TARGET_X = np.mean(centroid_x_tot) if centroid_x_tot else TARGET_X
#             TARGET_Y = np.mean(centroid_y_tot) if centroid_y_tot else TARGET_Y
#
#         if cx is not None and cy is not None:
#             if step_applied:
#                 error_x = TARGET_X - cx
#                 error_y = TARGET_Y - cy
#                 send_daq_signal(error_x, error_y)
#
#             centroid_x_tot.append(cx)
#             centroid_y_tot.append(cy)
#             time_list.append(elapsed_time)
#
#         if time.time() - start_time > 10:
#             processing = False
#             break
#
#     # Once processing is complete, plot the data
#     print(len(centroid_x_tot))
#     window.plot_centroid_data()

def process_frame(window):
    """ Tracks and corrects laser position using PID. Optimized for speed. """
    global processing, TARGET_X, TARGET_Y
    processing = True
    start_time = time.time()
    step_applied = False

    centroid_x_tot.clear()
    centroid_y_tot.clear()
    time_list.clear()

    while processing:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame  # ✅ Avoid unnecessary copying

        # 🔹 Convert to grayscale and apply threshold (Optimized)
        red_channel = frame[:, :, 2]
        _, red_thresh = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)

        # 🔹 Reduce noise with a **smaller** blur kernel (5x5 → 3x3)
        blurred = cv2.GaussianBlur(red_thresh, (3, 3), 0)

        # 🔹 Use OpenCV's `cv2.RETR_EXTERNAL` (only finds outermost contours)
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        if contours:
            # 🔹 Sort contours once instead of multiple times
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

            # 🔹 Optimize moment calculation (Avoid repeated function calls)
            centroids = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] > 100:  # ✅ Ignore tiny/noise contours (Threshold: 100 pixels)
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                    centroids.append((centroid_x, centroid_y))

            # Select the largest valid centroid
            if centroids:
                cx, cy = max(centroids, key=lambda c: c[1])  # Select lowest centroid in Y-axis

        # Step response logic
        elapsed_time = time.time() - start_time
        if elapsed_time > STEP_DELAY and not step_applied:
            step_applied = True
            TARGET_X = np.mean(centroid_x_tot) if centroid_x_tot else TARGET_X
            TARGET_Y = np.mean(centroid_y_tot) if centroid_y_tot else TARGET_Y

        if cx is not None and cy is not None:
            if step_applied:
                error_x = TARGET_X - cx
                error_y = TARGET_Y - cy
                send_daq_signal(error_x, error_y)

            centroid_x_tot.append(cx)
            centroid_y_tot.append(cy)
            time_list.append(elapsed_time)

        # ✅ Limit Processing Rate to **60 FPS** (or less)
        # time.sleep(1 / 60)

        if time.time() - start_time > 10:
            processing = False
            break

    # Plot data once processing is complete
    print(len(centroid_x_tot))
    print(len(time_list))
    window.plot_centroid_data()

class LaserTrackingApp(QWidget):
    """ GUI for live camera feed and laser tracking controls. """
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Timer for updating the video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(10)  # Refresh every 10ms

        # Start the frame capture thread
        self.capture_thread = threading.Thread(target=capture_frames, daemon=True)
        self.capture_thread.start()

    def init_ui(self):
        """ Initialize UI elements. """
        self.setWindowTitle("Laser Tracking System")
        self.setGeometry(100, 100, 700, 500)

        # Layout
        layout = QVBoxLayout()

        # Live video feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Buttons
        self.start_button = QPushButton("Start Processing", self)
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 14px;")
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing", self)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setStyleSheet("background-color: red; color: white; font-size: 14px;")
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

    def update_gui(self):
        """ Update the live camera feed in the GUI. """
        if latest_frame is not None:
            frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def start_processing(self):
        """ Start the frame processing in a separate thread. """
        global processing
        if not processing:
            processing_thread = threading.Thread(target=process_frame, args=(self,), daemon=True)
            processing_thread.start()

    def stop_processing(self):
        """ Stop the frame processing. """
        global processing
        processing = False

    def plot_centroid_data(self):
        """ Schedule plotting to run on the main thread """
        QTimer.singleShot(0, self._plot_data)

    def _plot_data(self):
        """ Plots centroid movement after processing stops. """
        plt.figure(figsize=(10, 5))
        print(len(time_list))
        print(len(centroid_x_tot))
        plt.plot(time_list, centroid_x_tot, label="X Position", color="blue")
        plt.plot(time_list, centroid_y_tot, label="Y Position", color="red")
        plt.axvline(x=STEP_DELAY, color='gray', linestyle='--', label="Correction Applied")
        plt.axhline(y=TARGET_X, color="blue", linestyle="dotted", label="Final Target X")
        plt.axhline(y=TARGET_Y, color="red", linestyle="dotted", label="Final Target Y")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Centroid Position (pixels)")
        plt.title("Laser Response to PID Correction")
        plt.legend()
        plt.grid()
        plt.show()

# Run the GUI
app = QApplication(sys.argv)
window = LaserTrackingApp()
window.show()
sys.exit(app.exec())