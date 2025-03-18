import threading

import cv2
import time
# Open the default webcam (0 represents the default camera)
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Disable auto-exposure (if supported)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Adjust manually (-6 to -2 is a good range)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def process_frame():
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fail at capture")
            continue

        # Extract only the red channel
        red_channel = frame[:, :, 2]

        # Threshold to isolate bright red areas (adjust threshold for different lighting conditions)
        _, red_thresh = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)

        # Find contours of red regions
        contours, _ = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (brightest red region)
            largest_contour = max(contours, key=cv2.contourArea)

            # Compute the centroid of the largest bright region
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw centroid on the original frame
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Green dot for centroid

                # Draw the contour of the brightest region
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)  # Yellow contour
        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Display results
        cv2.imshow("Webcam", frame)
        cv2.imshow("Thresholded", red_thresh)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# available_ports,working_ports,non_working_ports = list_ports()

thread = threading.Thread(process_frame(), daemon=True)
thread.start()

cap.release()
cv2.destroyAllWindows()

