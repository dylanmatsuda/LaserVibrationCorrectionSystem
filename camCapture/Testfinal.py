import sys
import threading
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from simple_pid import PID
from scipy.fft import fft, fftfreq
import nidaqmx
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFormLayout, QLineEdit, \
    QHBoxLayout, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QMetaObject

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
thresholded_frame = None  # ✅ Initialize this variable globally\
new_frame_available = False
running = True
processing = False  # Flag to track if processing is active

# Storage for Centroid Data
centroid_x_tot = []
centroid_y_tot = []
time_list = []

TARGET_X = 305
TARGET_Y = 235

# Step response test variables
STEP_DELAY = 5
CORRECTION_DURATION = 5

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

# 🔹 Global variable to store the oscillation frequency
oscillation_frequency = None

def send_daq_signal(window, error_x, error_y):
    """ Send correction signal using PID control """
    global voltage_x_tot, voltage_y_tot
    voltage_x_tot += pid_x(error_x)
    voltage_y_tot += pid_y(error_y)

    QMetaObject.invokeMethod(
        window,
        "update_voltage_labels",
        Qt.ConnectionType.QueuedConnection
    )

    # daq_task_x.write(voltage_x_tot)
    # daq_task_y.write(voltage_y_tot)

def capture_frames():
    """ Continuously captures frames for GUI display. """
    global latest_frame, running, frame_counter, new_frame_available
    while running:
        ret, frame = cap.read()
        if not ret:
            # time.sleep(0.1)
            continue
        with frame_lock:
            latest_frame = frame.copy()
            new_frame_available = True

def process_frame(window):
    """ Tracks and corrects laser position using PID. Optimized for speed. """
    global processing, TARGET_X, TARGET_Y, new_frame_available, latest_frame, thresholded_frame
    processing = True
    start_time = time.time()
    step_applied = False

    # Clear data from previous iterations
    centroid_x_tot.clear()
    centroid_y_tot.clear()
    time_list.clear()
    window.oscillation_frequency = None
    window.update_oscillation_label()

    while processing:
        if not new_frame_available:
            time.sleep(0.001)
            continue

        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()  # ✅ Copy frame to draw on
            new_frame_available = False

        # 🔹 Convert to grayscale and apply threshold
        red_channel = frame[:, :, 2]
        _, threshold = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)  # ✅ Store thresholded frame
        thresholded_frame = threshold.copy()

        # 🔹 Reduce noise with a **smaller** blur kernel (5x5 → 3x3)
        blurred = cv2.GaussianBlur(thresholded_frame, (3, 3), 0)
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

            centroids = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] > 100:
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                    centroids.append((centroid_x, centroid_y))

            if centroids:
                cx, cy = max(centroids, key=lambda c: c[1])

        # 🔹 Step Response Logic (Adjust Target Position After Delay)
        elapsed_time = time.time() - start_time
        if elapsed_time > STEP_DELAY and not step_applied:
            step_applied = True
            TARGET_X = np.mean(centroid_x_tot) if centroid_x_tot else TARGET_X
            TARGET_Y = np.mean(centroid_y_tot) if centroid_y_tot else TARGET_Y
            window.calculate_oscillation_frequency()  # ✅ Compute oscillation frequency
            window.update_correction_status("ACTIVE")   # Correction activated

        thresholded_bgr = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)

        # 🔹 Apply PID Correction & Store Data
        if cx is not None and cy is not None:
            if step_applied:
                error_x = TARGET_X - cx
                error_y = TARGET_Y - cy
                send_daq_signal(window, error_x, error_y)

            centroid_x_tot.append(cx)
            centroid_y_tot.append(cy)
            time_list.append(elapsed_time)

            # ✅ Draw Centroid and Text on **Thresholded Frame Only**
            cv2.circle(thresholded_bgr, (cx, cy), 8, (255, 255, 0), 2)  # ✅ White circle
            cv2.putText(thresholded_bgr, "Centroid", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        # ✅ Draw Dotted Crosshairs on `TARGET_X, TARGET_Y` (Only if Step Applied)
        if step_applied:
            frame_height, frame_width, _ = frame.shape

            # 🔹 Draw vertical dashed line at `TARGET_X`
            for i in range(0, frame_height, 10):
                if i % 20 == 0:
                    cv2.line(thresholded_bgr, (int(TARGET_X), i), (int(TARGET_X), i + 10), (255, 0, 255), 1)

            # 🔹 Draw horizontal dashed line at `TARGET_Y`
            for j in range(0, frame_width, 10):
                if j % 20 == 0:
                    cv2.line(thresholded_bgr, (j, int(TARGET_Y)), (j + 10, int(TARGET_Y)), (255, 0, 255), 1)

        # ✅ Lock and store latest processed frame for GUI update
        with frame_lock:
            latest_frame = frame.copy()
        with frame_lock:
            thresholded_frame = thresholded_bgr.copy()  # ✅ Prevent race conditions

        if time.time() - start_time > STEP_DELAY + CORRECTION_DURATION:
            processing = False
            break

    # Plot data once processing is complete
    print(len(centroid_x_tot))
    print(len(time_list))
    window.update_correction_status("IDLE")
    window.plot_centroid_data()

class ProcessingThread(QThread):
    position_updated = pyqtSignal(int, int)
    processing_done = pyqtSignal()
    time_elapsed = pyqtSignal(float)
    voltage_updated = pyqtSignal(float, float)  # ✅ Signal for Voltage
    correction_status_updated = pyqtSignal(str)  # ✅ Signal for Correction Status

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.running = True

    def run(self):
        global processing, TARGET_X, TARGET_Y, new_frame_available, latest_frame, thresholded_frame
        processing = True
        start_time = time.time()
        step_applied = False

        # Clear previous data
        centroid_x_tot.clear()
        centroid_y_tot.clear()
        time_list.clear()
        self.window.oscillation_frequency = None
        self.window.update_oscillation_label()

        while self.running:
            if not new_frame_available:
                QThread.yieldCurrentThread()
                continue

            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
                new_frame_available = False

            # 🔹 Convert to grayscale and apply threshold
            red_channel = frame[:, :, 2]
            _, threshold = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)
            thresholded_frame = threshold.copy()

            # 🔹 Reduce noise and find contours
            blurred = cv2.GaussianBlur(thresholded_frame, (3, 3), 0)
            contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cx, cy = None, None
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
                centroids = []
                for cnt in contours:
                    M = cv2.moments(cnt)
                    if M["m00"] > 100:
                        centroid_x = int(M["m10"] / M["m00"])
                        centroid_y = int(M["m01"] / M["m00"])
                        centroids.append((centroid_x, centroid_y))

                if centroids:
                    cx, cy = max(centroids, key=lambda c: c[1])

            # 🔹 Step Response Logic
            elapsed_time = time.time() - start_time
            self.time_elapsed.emit(elapsed_time)  # ✅ Emit elapsed time signal

            if elapsed_time > STEP_DELAY and not step_applied:
                step_applied = True
                TARGET_X = np.mean(centroid_x_tot) if centroid_x_tot else TARGET_X
                TARGET_Y = np.mean(centroid_y_tot) if centroid_y_tot else TARGET_Y
                self.window.calculate_oscillation_frequency()
                self.correction_status_updated.emit("ACTIVE")  # ✅ Emit correction status

            # 🔹 Apply PID Correction
            if cx is not None and cy is not None:
                if step_applied:
                    error_x = TARGET_X - cx
                    error_y = TARGET_Y - cy
                    voltage_x = pid_x(error_x)
                    voltage_y = pid_y(error_y)

                    global voltage_x_tot, voltage_y_tot
                    voltage_x_tot += voltage_x
                    voltage_y_tot += voltage_y

                    self.voltage_updated.emit(voltage_x_tot, voltage_y_tot)  # ✅ Emit voltage values

                centroid_x_tot.append(cx)
                centroid_y_tot.append(cy)
                time_list.append(elapsed_time)
                self.position_updated.emit(cx, cy)

            # ✅ Stop processing after correction duration
            if elapsed_time > STEP_DELAY + CORRECTION_DURATION:
                self.running = False
                self.quit()
                self.wait()
                break

        self.correction_status_updated.emit("IDLE")  # ✅ Emit correction status update
        self.processing_done.emit()

class LaserTrackingApp(QWidget):
    """ GUI for live camera feed and laser tracking controls. """
    def __init__(self):
        super().__init__()
        self.init_ui()

        self.oscillation_frequency = None  # Store frequency within the GUI

        self.processing_thread = None

        # Timer for updating the video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(10)  # Refresh every 10ms
        self.start_time = None  # Store the time when processing starts
        self.timer_running = False  # Flag to track if timer is running

        # Start the frame capture thread
        self.capture_thread = threading.Thread(target=capture_frames, daemon=True)
        self.capture_thread.start()

    def init_ui(self):
        """ Initialize UI elements with optimized layout. """
        self.setWindowTitle("Laser Tracking System")
        self.setGeometry(100, 100, 1920, 1080)  # Fullscreen window size

        # 🔹 **Main Vertical Layout**
        main_layout = QVBoxLayout()

        # 🔹 **Video Feeds Layout (Top Half, Full Width)**
        video_layout = QHBoxLayout()

        # 🔹 **Original Video Feed (Left)**
        video_frame = QVBoxLayout()
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_frame.addWidget(self.video_label)

        # 🔹 **Label for Original Video Feed**
        self.video_text = QLabel("Unprocessed Video Feed")
        self.video_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_text.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 5px;")
        video_frame.addWidget(self.video_text)

        video_layout.addLayout(video_frame)  # Add to row

        # 🔹 **Thresholded Video Feed with Overlays (Right)**
        threshold_frame = QVBoxLayout()
        self.threshold_label = QLabel(self)
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        threshold_frame.addWidget(self.threshold_label)

        # 🔹 **Label for Thresholded Video Feed**
        self.threshold_text = QLabel("Thresholded Video Feed with Overlays")
        self.threshold_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.threshold_text.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 5px;")
        threshold_frame.addWidget(self.threshold_text)

        video_layout.addLayout(threshold_frame)  # Add to row

        main_layout.addLayout(video_layout, 3)  # Assign more weight to video section

        # 🔹 **Bottom Layout (Controls + Data Display)**
        bottom_layout = QHBoxLayout()

        # 🔹 **Left Sidebar (Controls)**
        left_sidebar = QVBoxLayout()
        left_sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 🔹 **PID & Timing Inputs**
        pid_layout = QFormLayout()

        self.p_input = QLineEdit(str(pid_x.Kp))
        self.i_input = QLineEdit(str(pid_x.Ki))
        self.d_input = QLineEdit(str(pid_x.Kd))

        self.step_delay_input = QLineEdit(str(STEP_DELAY))
        self.correction_duration_input = QLineEdit(str(CORRECTION_DURATION))

        for input_field in [self.p_input, self.i_input, self.d_input, self.step_delay_input,
                            self.correction_duration_input]:
            input_field.setStyleSheet("font-size: 20px; padding: 5px; width: 100px;")

        self.p_input.returnPressed.connect(self.update_pid)
        self.i_input.returnPressed.connect(self.update_pid)
        self.d_input.returnPressed.connect(self.update_pid)
        self.step_delay_input.returnPressed.connect(self.update_timing)
        self.correction_duration_input.returnPressed.connect(self.update_timing)

        p_label = QLabel("P Gain:")
        i_label = QLabel("I Gain:")
        d_label = QLabel("D Gain:")
        step_label = QLabel("Step Delay (s):")
        correction_label = QLabel("Correction Duration (s):")

        for label in [p_label, i_label, d_label, step_label, correction_label]:
            label.setStyleSheet("font-size: 20px; font-weight: bold;")

        pid_layout.addRow(p_label, self.p_input)
        pid_layout.addRow(i_label, self.i_input)
        pid_layout.addRow(d_label, self.d_input)
        pid_layout.addRow(step_label, self.step_delay_input)
        pid_layout.addRow(correction_label, self.correction_duration_input)

        left_sidebar.addLayout(pid_layout)  # Add PID controls to left sidebar

        # 🔹 **Stacked Buttons**
        self.start_button = QPushButton("Start Processing", self)
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 20px; padding: 10px;")
        left_sidebar.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing", self)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setStyleSheet("background-color: red; color: white; font-size: 20px; padding: 10px;")
        left_sidebar.addWidget(self.stop_button)

        self.fullscreen_button = QPushButton("Toggle Fullscreen", self)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_button.setStyleSheet("font-size: 18px; padding: 8px;")
        left_sidebar.addWidget(self.fullscreen_button)

        bottom_layout.addLayout(left_sidebar, 1)  # Left (Controls)

        # 🔹 **Right Sidebar (Data Display)**
        right_sidebar = QVBoxLayout()
        right_sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 🔹 **Oscillation Frequency Display**
        self.oscillation_label = QLabel("Oscillation Frequency: N/A")
        self.oscillation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.oscillation_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            background-color: #f0f0f0;  /* Light Gray Background */
            border: 2px solid #888888;   /* Subtle Border */
            color: gray;
        """)
        right_sidebar.addWidget(self.oscillation_label)

        # 🔹 **Correction Status Indicator**
        self.correction_status_label = QLabel("Correction Status: IDLE")
        self.correction_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.correction_status_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 12px;
            border-radius: 8px;
            background-color: #f0f0f0;
            border: 2px solid #888888;
            color: gray;
        """)
        right_sidebar.addWidget(self.correction_status_label)

        # 🔹 **Voltage Output Display (X-Axis)**
        self.voltage_x_label = QLabel("X Voltage: 0.00 V")
        self.voltage_x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.voltage_x_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 12px;
            border-radius: 8px;
            background-color: #f0f0f0;
            border: 2px solid #888888;
            color: gray;
        """)
        right_sidebar.addWidget(self.voltage_x_label)

        # 🔹 **Voltage Output Display (Y-Axis)**
        self.voltage_y_label = QLabel("Y Voltage: 0.00 V")
        self.voltage_y_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.voltage_y_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 12px;
            border-radius: 8px;
            background-color: #f0f0f0;
            border: 2px solid #888888;
            color: gray;
        """)
        right_sidebar.addWidget(self.voltage_y_label)

        bottom_layout.addLayout(right_sidebar, 1)  # Right (Data Display)

        main_layout.addLayout(bottom_layout, 1)  # Assign less weight to bottom section

        # 🔹 **Time Elapsed Counter**
        self.time_elapsed_label = QLabel("Time Elapsed: 0.00s")
        self.time_elapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_elapsed_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 12px;
            border-radius: 8px;
            background-color: #f0f0f0;
            border: 2px solid #888888;
            color: gray;
        """)
        right_sidebar.addWidget(self.time_elapsed_label)

        self.setLayout(main_layout)
        self.showFullScreen()  # Auto fullscreen

    def update_gui(self):
        """ Update both video feeds in the GUI. """
        if latest_frame is not None:
            frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))  # ✅ Original video feed

        if thresholded_frame is not None:
            if len(thresholded_frame.shape) == 2:  # If it's grayscale, convert to BGR
                thresholded_bgr = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)
            else:
                thresholded_bgr = thresholded_frame  # Already BGR

            frame_rgb = cv2.cvtColor(thresholded_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB for Qt
            h, w, ch = frame_rgb.shape
            qimg_thresh = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.threshold_label.setPixmap(QPixmap.fromImage(qimg_thresh))  # ✅ Show updated feed

    def toggle_fullscreen(self):
        """ Toggle between fullscreen and windowed mode. """
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def update_pid(self):
        """ Update PID values when the user presses Enter. """
        try:
            new_p = float(self.p_input.text())
            new_i = float(self.i_input.text())
            new_d = float(self.d_input.text())

            # Update global PID controllers
            pid_x.Kp = new_p
            pid_x.Ki = new_i
            pid_x.Kd = new_d

            if new_p <= 0 or new_i < 0 or new_d < 0:
                raise ValueError("PID input cannot be negative! Please enter a valid float.")

            print(f"Updated PID: P={new_p}, I={new_i}, D={new_d}")

        except ValueError:
            raise ValueError("Invalid PID input! Please enter a valid float.")

    def update_timing(self):
        """ Update Step Delay & Correction Duration when user presses Enter. """
        global STEP_DELAY, CORRECTION_DURATION
        try:
            new_step_delay = float(self.step_delay_input.text())
            new_correction_duration = float(self.correction_duration_input.text())

            STEP_DELAY = new_step_delay
            CORRECTION_DURATION = new_correction_duration

            if STEP_DELAY < 0 or CORRECTION_DURATION < 0:
                raise ValueError("Invalid input! Please enter a positive valid float for Step Delay and Correction Duration.")

            print(f"Updated Timing: STEP_DELAY={STEP_DELAY}s, CORRECTION_DURATION={CORRECTION_DURATION}s")

        except ValueError:
            raise ValueError("Invalid input! Please enter a valid float for Step Delay and Correction Duration.")

    def calculate_oscillation_frequency(self):
        """ Compute the dominant oscillation frequency using FFT. """
        global oscillation_frequency

        if len(time_list) < 2 or len(centroid_x_tot) < 2:
            oscillation_frequency = 0  # Not enough data
            return

        # 🔹 Compute Sample Rate (Time between each frame)
        time_intervals = np.diff(time_list)  # Time differences
        avg_sample_rate = 1 / np.mean(time_intervals)  # In Hz

        # 🔹 FFT Calculation
        N = len(centroid_x_tot)  # Number of data points
        fft_values = fft(centroid_x_tot - np.mean(centroid_x_tot))  # Remove DC component
        frequencies = fftfreq(N, d=1 / avg_sample_rate)  # Frequency axis

        # 🔹 Find Dominant Frequency (Ignoring DC Component)
        magnitude_spectrum = np.abs(fft_values[:N // 2])  # Only positive frequencies
        dominant_index = np.argmax(magnitude_spectrum[1:]) + 1  # Ignore DC (index 0)
        self.oscillation_frequency = frequencies[dominant_index]

        self.update_oscillation_label()

    def update_oscillation_label(self):
        """ Updates the oscillation frequency display with dynamic styling. """
        if self.oscillation_frequency is None:
            self.oscillation_label.setText("Oscillation Frequency: Calculating...")
            self.oscillation_label.setStyleSheet("""
                font-size: 24px;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                background-color: #f0f0f0;
                border: 2px solid #888888;
                color: gray;
            """)
        else:
            self.oscillation_label.setText(f"Oscillation Frequency: {self.oscillation_frequency:.2f} Hz")
            self.oscillation_label.setStyleSheet("""
                font-size: 24px;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                background-color: #d4edda;  /* Light Green for Success */
                border: 2px solid #155724;
                color: #155724;  /* Dark Green Text */
            """)  # ✅ Changes color to **green** when frequency is detected

    def update_correction_status(self, status):
        """ Updates correction status dynamically. """
        color = "#155724" if status == "ACTIVE" else "gray"
        background = "#d4edda" if status == "ACTIVE" else "#f0f0f0"

        self.correction_status_label.setText(f"Correction Status: {status}")
        self.correction_status_label.setStyleSheet(f"""
            font-size: 22px;
            font-weight: bold;
            padding: 12px;
            border-radius: 8px;
            background-color: {background};
            border: 2px solid {color};
            color: {color};
        """)

    def update_voltage_labels(self, voltage_x, voltage_y):
        """ Updates voltage display using signal. """
        self.voltage_x_label.setText(f"X Voltage: {voltage_x:.2f} V")
        self.voltage_y_label.setText(f"Y Voltage: {voltage_y:.2f} V")

    def update_time_elapsed(self, elapsed_time):
        """ Updates the elapsed time counter in real-time. """
        self.time_elapsed_label.setText(f"Time Elapsed: {elapsed_time:.2f}s")

    def start_processing(self):
        """ Start processing using QThread to avoid GUI freezing. """
        global processing
        if not processing:
            try:
                self.update_pid()
                self.update_timing()
            except ValueError as e:
                self.show_error_message(f"Input Error: {str(e)}")
                return

            self.start_button.setEnabled(False)
            processing = True

            # ✅ Start Processing Thread and Connect Signals
            self.processing_thread = ProcessingThread(self)
            self.processing_thread.position_updated.connect(self.update_position_display)
            self.processing_thread.processing_done.connect(self.process_complete)
            self.processing_thread.time_elapsed.connect(self.update_time_elapsed)
            self.processing_thread.voltage_updated.connect(self.update_voltage_labels)  # ✅ New Signal
            self.processing_thread.correction_status_updated.connect(self.update_correction_status)  # ✅ New Signal
            self.processing_thread.start()

    def process_complete(self):
        """ Stops the processing thread and re-enables UI controls. """
        global processing
        processing = False  # 🔹 Stop processing flag

        # 🔹 Stop QThread
        if self.processing_thread:
            self.processing_thread.running = False
            self.processing_thread.quit()
            self.processing_thread.wait()

        self.start_button.setEnabled(True)  # 🔹 Re-enable start button
        self.timer_running = False  # 🔹 Stop time counter

    def process_frame_with_ui_update(self):
        """ Runs process_frame and re-enables the button when complete. """
        process_frame(self)  # 🔹 Run the processing function
        global processing
        processing = False  # 🔹 Ensure processing is marked as stopped

        # 🔹 **Stop Timer**
        self.timer_running = False

        QTimer.singleShot(0, self.enable_start_button)  # 🔹 Re-enable button safely

    def enable_start_button(self):
        """ Re-enable the Start Processing button after processing completes. """
        self.start_button.setEnabled(True)

    def stop_processing(self):
        """ Stop the frame processing. """
        global processing
        processing = False

    def show_error_message(self, message):
        """ Displays an error alert box with the given message. """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Input Error")
        msg.setText(message)
        msg.exec()

    def plot_centroid_data(self):
        """ Schedule plotting to run on the main thread """
        QTimer.singleShot(0, self._plot_data)

    def _plot_data(self):
        """ Plots centroid movement after processing stops. """
        plt.figure(figsize=(10, 5))
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