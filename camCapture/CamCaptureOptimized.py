import threading
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from simple_pid import PID

import nidaqmx

# Open the webcam (try different indexes if 2 doesn't work)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Use this if testing with laptop webcam
# cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 60)

# Try to disable auto-exposure and manually set exposure (if supported)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Adjust manually (-6 to -2 is a good range)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Thread-safe frame storage
frame_lock = threading.Lock()
latest_frame = None
running = True  # Global flag to control both threads

# Storage for Centroid Data and Frequency
centroid_x = []  # Stores x-coordinates of the centroid
centroid_y = []  # Stores y-coordinates of the centroid
centroid_timestamps = []  # Stores timestamps of centroid calculations
centroid_freq_list = []  # Stores centroid frequency over time
time_list = []  # Stores time elapsed for plotting

# Define target centroid position (adjust as needed)
TARGET_X = 320  # Assume center of the image
TARGET_Y = 240

# DAQ setup
daq_task_x = nidaqmx.Task()
daq_task_y = nidaqmx.Task()

daq_task_x.ao_channels.add_ao_voltage_chan("Dev1/ao0")  # X-channel
daq_task_y.ao_channels.add_ao_voltage_chan("Dev1/ao1")  # Y-channel


# PID controllers for X and Y corrections
# pid_x = PID(0.001, 0.0005, 0.05, setpoint=TARGET_X)  # Tune gains as needed
pid_x = PID(0, 0, 0, setpoint=TARGET_X)  # Tune gains as needed
pid_y = PID(0.001, 0.0005, 0.1, setpoint=TARGET_Y)
pid_x.output_limits = (-8, 8)  # Voltage range for DAQ
pid_y.output_limits = (-8, 8)

def send_daq_signal(error_x, error_y):
    """
    Computes the correction voltage dynamically using PID control.
    """
    voltage_x = pid_x(error_x)
    voltage_y = pid_y(error_y)

    # Send corrected signal to DAQ
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


# def process_frame():
#     """ Process the frame, calculate FPS and centroid frequency, and display the results. """
#     global running
#     prev_time = time.time()
#     start_time = time.time()  # Start time for plotting
#
#     while running:
#         with frame_lock:
#             if latest_frame is None:
#                 continue
#             frame = latest_frame.copy()
#
#         # Convert to red channel and apply blur to reduce noise
#         red_channel = frame[:, :, 2]
#         blurred = cv2.GaussianBlur(red_channel, (5, 5), 0)
#
#         # Thresholding for bright red areas
#         _, red_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
#
#         # Find contours in the thresholded image
#         contours, _ = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         cx, cy = None, None  # Default to None in case no centroid is found
#
#         if contours:
#             # Find the largest contour (brightest red region)
#             largest_contour = max(contours, key=cv2.contourArea)
#
#             # Compute the centroid of the largest bright region
#             M = cv2.moments(largest_contour)
#             if M["m00"] > 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#
#                 # Draw centroid on the original frame
#                 cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Green dot for centroid
#                 cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)  # Yellow contour
#
#
#                 # Store timestamp of centroid calculation
#                 centroid_timestamps.append(time.time())
#
#         # FPS Calculation (Moving Average)
#         curr_time = time.time()
#         frame_time = curr_time - prev_time
#         prev_time = curr_time
#
#         # Compute Centroid Frequency (Hz)
#         if len(centroid_timestamps) > 1:
#             time_diff = centroid_timestamps[-1] - centroid_timestamps[0]
#             centroid_freq = (len(centroid_timestamps) - 1) / time_diff if time_diff > 0 else 0
#         else:
#             centroid_freq = 0
#
#         # Store centroid coordinates
#         if not centroid_x:
#             centroid_x.append(0)
#             centroid_y.append(0)
#         else:
#             centroid_x.append(cx)
#             centroid_y.append(cy)
#         # Store data for plotting
#         elapsed_time = curr_time - start_time  # Time since start of recording
#         centroid_freq_list.append(centroid_freq)
#         time_list.append(elapsed_time)
#
#         # Display FPS and Centroid Frequency on frame
#         cv2.putText(frame, f"Centroid Hz: {centroid_freq:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#         # Display results
#         cv2.imshow("Webcam", frame)
#         cv2.imshow("Thresholded", red_thresh)
#
#         # Press 'q' to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             running = False  # Stop both threads
#             break
#
#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()

# def process_frame():
#     """ Track oscillating laser spot and dynamically correct tip-tilt mirror. """
#     global running
#     prev_time = time.time()
#     start_time = time.time()
#     prev_positions_x = []  # Stores recent X values for frequency estimation
#     prev_positions_y = []  # Stores recent Y values
#
#     while running:
#         with frame_lock:
#             if latest_frame is None:
#                 continue
#             frame = latest_frame.copy()
#
#         # Convert to red channel and process image
#         red_channel = frame[:, :, 2]
#         blurred = cv2.GaussianBlur(red_channel, (5, 5), 0)
#         _, red_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
#
#         # Find contours
#         contours, _ = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         cx, cy = None, None
#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
#             M = cv2.moments(largest_contour)
#             if M["m00"] > 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#
#                 cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
#                 cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)
#
#             # Store recent positions to estimate oscillation frequency
#                 prev_positions_x.append(cx)
#                 prev_positions_y.append(cy)
#
#                 if len(prev_positions_x) > 10:
#                     prev_positions_x.pop(0)
#                     prev_positions_y.pop(0)
#
#                 # Compute error
#                 error_x = TARGET_X - cx
#                 error_y = TARGET_Y - cy
#
#                 # Dynamically correct mirror position
#                 send_daq_signal(error_x, error_y)
#
#                 centroid_timestamps.append(time.time())
#
#             # Store centroid coordinates
#                 if not centroid_x:
#                     centroid_x.append(0)
#                     centroid_y.append(0)
#                 else:
#                     centroid_x.append(cx)
#                     centroid_y.append(cy)
#             curr_time = time.time()
#             elapsed_time = curr_time - start_time  # Time since start of recording
#             time_list.append(elapsed_time)
#
#         # Display frame
#         cv2.putText(frame, f"Centroid Hz: {len(centroid_timestamps)/(time.time()-start_time):.2f}",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.imshow("Webcam", frame)
#         cv2.imshow("Thresholded", red_thresh)
#
#         # Stop on 'q' press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             running = False
#             break
#
#     daq_task_x.close()
#     daq_task_y.close()
#     cap.release()
#     cv2.destroyAllWindows()

def plot_centroid_data():
    """ Plots centroid values and frequency over time after data collection. """
    fig, axs = plt.subplots(2, figsize=(10, 8))

    # Plot centroid movement over time
    axs[0].plot(time_list[0:len(centroid_x)], centroid_x, label="Centroid X", color="blue", linestyle="-")
    # axs[0].plot(time_list[0:len(centroid_x)], centroid_y, label="Centroid Y", color="red", marker="o", linestyle="-")
    axs[0].set_xlabel("Time (seconds)")
    axs[0].set_ylabel("Centroid Position (pixels)")
    axs[0].set_title("Centroid X Movement Over Time")
    axs[0].legend()
    axs[0].grid(True)

    # axs[0].plot(time_list[0:len(centroid_x)], centroid_x, label="Centroid X", color="blue", marker="o", linestyle="-")
    axs[1].plot(time_list[0:len(centroid_y)], centroid_y, label="Centroid Y", color="red", linestyle="-")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Centroid Position (pixels)")
    axs[1].set_title("Centroid Y Movement Over Time")
    axs[1].legend()
    axs[1].grid(True)

    # # Plot centroid frequency over time
    # axs[1].plot(time_list, centroid_freq_list, label="Centroid Frequency (Hz)", color="green", marker="o", linestyle="-")
    # axs[1].set_xlabel("Time (seconds)")
    # axs[1].set_ylabel("Frequency (Hz)")
    # axs[1].set_title("Centroid Calculation Frequency Over Time")
    # axs[1].legend()
    # axs[1].grid(True)

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
print(len(centroid_y)/time_list[-1])

print("Program exited cleanly.")
