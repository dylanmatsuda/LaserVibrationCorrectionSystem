import threading
from contextlib import nullcontext

import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from simple_pid import PID

import nidaqmx

# Open the webcam (try different indexes if 2 doesn't work)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Use this if testing with laptop webcam
# cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FPS, 60)

# Try to disable auto-exposure and manually set exposure (if supported)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Adjust manually (-6 to -2 is a good range)
cap.set(cv2.CAP_PROP_GAIN, -6)  # Adjust manually (-6 to -2 is a good range)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.1)

#
# time.sleep(2)  # Allow camera to initialize
#
# # Try multiple methods of disabling auto-exposure
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Some cameras use 0
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)  # Some cameras use 1.0
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Some cameras require 3 to enable manual mode
#
# time.sleep(1)
#
# # Set exposure multiple times to ensure it applies
# for i in range(5):  # Apply exposure multiple times for stubborn cameras
#     cap.set(cv2.CAP_PROP_EXPOSURE, -6)
#     time.sleep(0.5)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Thread-safe frame storage
frame_lock = threading.Lock()
latest_frame = None
running = True  # Global flag to control both threads

# Storage for Centroid Data and Frequency
centroid_x_tot= []  # Stores x-coordinates of the centroid
centroid_y_tot = []  # Stores y-coordinates of the centroid
centroid_timestamps = []  # Stores timestamps of centroid calculations
centroid_freq_list = []  # Stores centroid frequency over time
time_list = []  # Stores time elapsed for plotting

# Define target centroid position (adjust as needed)
TARGET_X = 430  # Assume center of the image
TARGET_Y = 179

# DAQ setup
daq_task_x = nidaqmx.Task()
daq_task_y = nidaqmx.Task()

daq_task_x.ao_channels.add_ao_voltage_chan("Dev1/ao0")  # X-channel
daq_task_y.ao_channels.add_ao_voltage_chan("Dev1/ao1")  # Y-channel


# PID controllers for X and Y corrections
# pid_x = PID(0.001, 0.0005, 0.05, setpoint=TARGET_X)  # Tune gains as needed
pid_x = PID(0, 0, 0, setpoint=TARGET_X)  # Tune gains as needed
pid_y = PID(0.001, 0.0005, 0.001, setpoint=TARGET_Y)
pid_x.output_limits = (-8, 8)  # Voltage range for DAQ
pid_y.output_limits = (-10, 10)

def send_daq_signal(error_x, error_y):
    """
    Computes the correction voltage dynamically using PID control.
    """
    voltage_x = pid_x(-error_x)
    voltage_y = pid_y(-error_y)

    # Send corrected signal to DAQ
    # print(voltage_y)
    daq_task_x.write(voltage_x)
    daq_task_y.write(voltage_y)

def capture_frames():
    """ Continuously captures frames in a separate thread to avoid blocking processing. """
    global latest_frame, running
    while running:
        ret, frame = cap.read()

        if not ret:
            print("⚠️ Warning: Failed to capture frame. Retrying...")
            time.sleep(0.1)  # Short delay to prevent excessive retries
            continue  # Try again

        with frame_lock:
            latest_frame = frame.copy()

import threading
import cv2
import time
import numpy as np

# Define target position (center of the frame)
TARGET_X = 320  # Adjust based on your setup
TARGET_Y = 450  # Adjust based on your setup

def process_frame():
    """ Track oscillating laser spots and dynamically correct tip-tilt mirror. """
    global running
    prev_time = time.time()
    start_time = time.time()
    prev_positions_x = []  # Stores recent X values for frequency estimation
    prev_positions_y = []  # Stores recent Y values

    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # Convert to red channel and process image
        red_channel = frame[:, :, 2]
        blurred = cv2.GaussianBlur(red_channel, (5, 5), 0)
        _, red_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None  # Default centroid
        if len(contours) > 0:
            # Sort contours by area and get the two largest
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

            centroids = []

            centroid_x = None
            centroid_y = None

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                    # distance = np.sqrt((centroid_x - TARGET_X) ** 2 + (centroid_y - TARGET_Y) ** 2)
                    distance = centroid_y - TARGET_Y
                    centroids.append((centroid_x, centroid_y, distance))

            # Select the centroid farthest from the center
            if centroids:
                cx, cy, _ = max(centroids, key=lambda c: c[2])

                # Draw both centroids
                for c in centroids:
                    cv2.circle(frame, (c[0], c[1]), 5, (255, 0, 0), -1)  # Blue dot for second largest
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Green dot for main centroid
                cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)  # Yellow outline for selected

                # Store recent positions to estimate oscillation frequency
                prev_positions_x.append(cx)
                prev_positions_y.append(cy)

                if len(prev_positions_x) > 10:
                    prev_positions_x.pop(0)
                    prev_positions_y.pop(0)

                # Compute error
                error_x = TARGET_X - cx
                error_y = TARGET_Y - cy
                # print(error_y)
                print("x:" + str(cx))
                print("y:" + str(cy))

                # Dynamically correct mirror position
                # send_daq_signal(error_x, error_y)

                centroid_timestamps.append(time.time())

            # Store centroid coordinates
            if not centroid_x:
                centroid_x_tot.append(0)
                centroid_y_tot.append(0)
            else:
                centroid_x_tot.append(cx)
                centroid_y_tot.append(cy)

            curr_time = time.time()
            elapsed_time = curr_time - start_time  # Time since start of recording
            time_list.append(elapsed_time)

        # Display frame
        cv2.putText(frame, f"Centroid Hz: {len(centroid_timestamps)/(time.time()-start_time):.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Webcam", frame)
        cv2.imshow("Thresholded", red_thresh)

        # Stop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    daq_task_x.close()
    daq_task_y.close()
    cap.release()
    cv2.destroyAllWindows()

def plot_centroid_data():
    """ Plots centroid values and frequency over time after data collection. """
    fig, axs = plt.subplots(2, figsize=(10, 8))

    # Plot centroid movement over time
    axs[0].plot(time_list[0:len(centroid_x_tot)], centroid_x_tot, label="Centroid X", color="blue", linestyle="-")
    # axs[0].plot(time_list[0:len(centroid_x)], centroid_y, label="Centroid Y", color="red", marker="o", linestyle="-")
    axs[0].set_xlabel("Time (seconds)")
    axs[0].set_ylabel("Centroid Position (pixels)")
    axs[0].set_title("Centroid X Movement Over Time")
    axs[0].legend()
    axs[0].grid(True)

    print(len(centroid_x_tot))
    print(len(time_list))

    # axs[0].plot(time_list[0:len(centroid_x)], centroid_x, label="Centroid X", color="blue", marker="o", linestyle="-")
    axs[1].plot(time_list[0:len(centroid_y_tot)], centroid_y_tot, label="Centroid Y", color="red", linestyle="-")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Centroid Position (pixels)")
    axs[1].set_title("Centroid Y Movement Over Time")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# Start threads
capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_frame, daemon=True)

capture_thread.start()
process_thread.start()

# Keep the main thread alive
while running:
    try:
        time.sleep(1)  # Prevents CPU overuse
    except KeyboardInterrupt:
        running = False  # Allow manual exit with Ctrl+C

# Ensure threads terminate before exiting
capture_thread.join()
process_thread.join()

# Plot centroid movement and frequency after the program stops
plot_centroid_data()
# print(len(centroid_y)/time_list[-1])

print("Program exited cleanly.")
