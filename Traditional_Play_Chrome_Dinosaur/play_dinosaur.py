import cv2
import numpy as np
import pyautogui
from collections import deque
import threading
import time
import os

# Run Chrome Dinosaur game
def start_dino_game():
    try:
        os.system("python3 dino_main.py")
    except Exception as e:
        print("Failure:", e)

threading.Thread(target=start_dino_game, daemon=True).start()
time.sleep(2)

# Parameters
FRAME_HEIGHT, FRAME_WIDTH = 400, 600
region_top, region_bottom = 0, int(2 * FRAME_HEIGHT / 3)
region_left, region_right = int(FRAME_WIDTH / 2), FRAME_WIDTH
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18
frames_elapsed = 0

background = None
hand = None
gesture_buffer = deque(maxlen=3)
current_action = "Idle"

# Hand Data
class HandData:
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        self.fingers = 0
        self.gestureList = []
        self.isInFrame = True

    def update(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX

# Region of Interest

def get_region(frame):
    region = frame[region_top:region_bottom, region_left:region_right]
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = cv2.GaussianBlur(region, (5, 5), 0)
    return region

# Background Modeling

def get_average(region):
    global background
    if background is None:
        background = region.copy().astype("float")
    else:
        cv2.accumulateWeighted(region, background, BG_WEIGHT)

# Background Segmentation

def segment(region):
    global background
    diff = cv2.absdiff(background.astype(np.uint8), region)
    thresholded = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresholded = cv2.medianBlur(thresholded, 5)
    thresholded = cv2.erode(thresholded, None, iterations=1)
    thresholded = cv2.dilate(thresholded, None, iterations=2)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, thresholded
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 1000:
        return None, thresholded
    return max_contour, thresholded

# Counting fingers Methods

def count_fingers(thresh, hand):
    # Use Convex Hull + Convexity Defects to count
    contour, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contour:
        return 0

    contour = max(contour, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=False)

    if len(hull) < 3:
        return 0

    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Use lengths of triangle to calculate angle
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c + 1e-5))  # Avoid zero fraction

        # When angle is less than 90, we recognize it as the real finger crack
        if angle <= np.pi / 2 and d > 10000:  # d = depth, acoid shallow cracks are recognized as fingers
            count += 1

    return count + 1 if count > 0 else 0

# Multi-frame stabilization

def most_frequent(lst):
    return max(set(lst), key=lst.count) if lst else 0

# Execution
# We found the method "press" is not good, key down for a while can make sure the actions are correctly executed
def hold_up_key():
    pyautogui.keyDown('up')
    time.sleep(0.15)  # Less than down
    pyautogui.keyUp('up')

def hold_down_key():
    pyautogui.keyDown('down')
    time.sleep(0.8)
    pyautogui.keyUp('down')

def perform_action(fingers):
    global current_action
    gesture_buffer.append(fingers)
    action = most_frequent(gesture_buffer)

    if action == 4:
        threading.Thread(target=hold_up_key, daemon=True).start()
        current_action = "Jump"
    elif action == 2:
        threading.Thread(target=hold_down_key, daemon=True).start()
        current_action = "Duck"
    else:
        current_action = "Idle"

# GUI output word

def write_on_image(frame):
    cv2.putText(frame, f"Gesture: {current_action}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)

# Main loop
cap = cv2.VideoCapture(0)
print("Hand control is running, press 'q' to quit the idle...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)

    region = get_region(frame)

    if frames_elapsed < CALIBRATION_TIME:
        get_average(region)
        cv2.putText(frame, "Calibrating...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        contour, thresholded = segment(region)
        if contour is not None:
            hull = cv2.convexHull(contour)
            top = tuple(hull[hull[:, :, 1].argmin()][0])
            bottom = tuple(hull[hull[:, :, 1].argmax()][0])
            left = tuple(hull[hull[:, :, 0].argmin()][0])
            right = tuple(hull[hull[:, :, 0].argmax()][0])
            centerX = (left[0] + right[0]) // 2
            if hand is None:
                hand = HandData(top, bottom, left, right, centerX)
            else:
                hand.update(top, bottom, left, right, centerX)

            fingers = count_fingers(thresholded, hand)
            perform_action(fingers)
            cv2.drawContours(thresholded, [contour], -1, (255, 255, 255), 2)
            cv2.imshow("Mask", thresholded)
        else:
            current_action = "Idle"

    write_on_image(frame)
    cv2.imshow("Gesture Control", frame)
    frames_elapsed += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
