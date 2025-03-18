import threading
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from simple_pid import PID
import nidaqmx

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_GAIN, -6)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Thread-safe frame storage
frame_lock = threading.Lock()
latest_frame = None
running = True  # Global flag

# Storage for Centroid Data and Frequency
centroid_x_tot = []
centroid_y_tot = []
time_list = []

# Define target centroid position
TARGET_X = 305  # Assume center of the image
TARGET_Y = 235

# Step response test variables
STEP_DELAY = 5  # Seconds before applying the step change
STEP_AMOUNT = -50  # Pixels to shift setpoint

# DAQ setup
daq_task_x = nidaqmx.Task()
daq_task_y = nidaqmx.Task()
daq_task_x.ao_channels.add_ao_voltage_chan("Dev1/ao0")
daq_task_y.ao_channels.add_ao_voltage_chan("Dev1/ao1")

# PID Controllers
# pid_x = PID(0, 0, 0, setpoint=TARGET_X)
pid_x = PID(0.0045, 0, 0.000, setpoint=0)
pid_y = PID(0, 0, 0, setpoint=TARGET_Y)
# pid_y = PID(0.01, 0, 0, setpoint=TARGET_Y)
pid_x.output_limits = (-10, 10)
pid_y.output_limits = (-10, 10)

voltage_x_tot = 0
voltage_y_tot = 0


def send_daq_signal(error_x, error_y):
    global voltage_x_tot

    voltage_x_tot += pid_x(error_x)

    # voltage_x_tot -= error_x/10000

    print(error_x)
    print("pid output:" + str(pid_x(error_x)))
    voltage_y = pid_y(error_y)
    daq_task_x.write(voltage_x_tot)
    daq_task_y.write(voltage_y)
    # print(voltage_y)


def capture_frames():
    """ Continuously captures frames in a separate thread to avoid blocking processing. """
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Failed to capture frame. Retrying...")
            time.sleep(0.1)
            continue
        with frame_lock:
            latest_frame = frame.copy()


def process_frame():
    """ Track laser spots and dynamically correct mirror position. """
    global running, TARGET_X, TARGET_Y
    start_time = time.time()
    step_applied = False  # Track whether the step change has been applied
    # step_applied = True  # Track whether the step change has been applied


    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # Process Image
        red_channel = frame[:, :, 2]
        blurred = cv2.GaussianBlur(red_channel, (5, 5), 0)
        _, red_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

            centroids = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                    distance = centroid_y - TARGET_Y
                    centroids.append((centroid_x, centroid_y, distance))

            if centroids:
                cx, cy, _ = max(centroids, key=lambda c: c[2])

                for c in centroids:
                    cv2.circle(frame, (c[0], c[1]), 5, (255, 0, 0), -1)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)

                # Apply Step Response Test
        elapsed_time = time.time() - start_time
        if elapsed_time > STEP_DELAY and not step_applied:
            step_applied = True
            print(f"Applying correction at {elapsed_time:.2f} sec")
            TARGET_X = np.mean(centroid_x_tot)
            TARGET_Y = np.mean(centroid_y_tot)
            print("TARGET_X: " + str(TARGET_X))
            print("TARGET_Y: " + str(TARGET_Y))

        #     TARGET_X += STEP_AMOUNT
        #     TARGET_Y += STEP_AMOUNT
        #     step_applied = True

            # Compute Error
        if cx is not None and cy is not None:
            if step_applied:
                error_x = TARGET_X - cx
                error_y = TARGET_Y - cy
                send_daq_signal(error_x, error_y)

            # Store centroid data
            centroid_x_tot.append(cx)
            centroid_y_tot.append(cy)
            time_list.append(elapsed_time)

        # Display Frame
        cv2.putText(frame, f"Centroid Hz: {len(centroid_x_tot) / (time.time() - start_time):.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Webcam", frame)
        cv2.imshow("Thresholded", red_thresh)

        if time.time() - start_time > 10:
            running = False
            break

        # Stop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break



    daq_task_x.close()
    daq_task_y.close()
    cap.release()
    cv2.destroyAllWindows()


def plot_centroid_data():
    """ Plots centroid movement during step response test with final target position. """
    plt.figure(figsize=(10, 5))

    # Plot centroid positions
    plt.plot(time_list, centroid_x_tot, label="X Position", color="blue")
    plt.plot(time_list, centroid_y_tot, label="Y Position", color="red")

    # Add vertical line for step change
    plt.axvline(x=STEP_DELAY, color='gray', linestyle='--', label="Correction Applied")

    # Add horizontal dotted lines for final target positions
    plt.axhline(y=TARGET_X, color="blue", linestyle="dotted", label="Final Target X")
    plt.axhline(y=TARGET_Y, color="red", linestyle="dotted", label="Final Target Y")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Centroid Position (pixels)")
    plt.title("Laser Response to PID Correction")
    plt.legend()
    plt.grid()
    plt.show()


# Start threads
capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_frame, daemon=True)

capture_thread.start()
process_thread.start()

# Keep main thread alive
while running:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        running = False

    # Ensure threads terminate before exiting
capture_thread.join()
process_thread.join()

# Plot centroid movement and step response
plot_centroid_data()

print("Program exited cleanly.")
