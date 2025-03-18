import cv2
import time
from cv2_enumerate_cameras import enumerate_cameras

# for camera_info in enumerate_cameras(cv2.CAP_MSMF):
#     print(f'{camera_info.index}: {camera_info.name}')
#
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Disable auto exposure
# cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Set exposure manually
#
# # Verify if the exposure was set
# print("Auto Exposure:", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
# print("Exposure:", cap.get(cv2.CAP_PROP_EXPOSURE))
#
# cap.release()

cap.release()  #  the camera
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Reopen camera
time.sleep(2)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
