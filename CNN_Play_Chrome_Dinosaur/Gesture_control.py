import cv2
import numpy as np
import os
import time
import threading
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import pyautogui
import subprocess

# Load CNN Gesture Detection Model
def CNN_model():
    input_shape = (200, 200, 1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

model = CNN_model()
model.load_weights("./gesture_model.h5")

# Define Gesture Category
gesture_labels = ["NOTHING", "STOP", "PEACE", "PUNCH"]

# Set ROI (Region of Interest)
x0, y0, width, height = 400, 200, 200, 200

# Prediction model
guessGestureMode = False

# Record recentest gesture and time
last_gesture = None
gesture_count = 0  # Record continuous same gesture number
last_action_time = 0  # Last press time


# Bi-value Graph
def binaryMask(frame, x0, y0, width, height):
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 2)
    
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    return res


def jump_action():
    pyautogui.keyDown("up")
    time.sleep(0.2)
    pyautogui.keyUp("up")

def duck_action():
    pyautogui.keyDown("down")
    time.sleep(0.8)
    pyautogui.keyUp("down")

def guessGesture(model, roi):
    global last_gesture, gesture_count, last_action_time, current_action

    img = cv2.resize(roi, (200, 200))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    preds = model.predict(img)
    label_index = np.argmax(preds)
    confidence = preds[0][label_index] * 100
    gesture = gesture_labels[label_index]

    if confidence < 50:
        current_action = "Idle"
        return "NOTHING", confidence

    if gesture == last_gesture:
        gesture_count += 1
    else:
        gesture_count = 1
        last_gesture = gesture

    if gesture_count >= 2 and time.time() - last_action_time > 0.3:
        last_action_time = time.time()

        if gesture == "STOP":
            threading.Thread(target=jump_action, daemon=True).start()
            current_action = "Jump"
        elif gesture == "PEACE":
            threading.Thread(target=duck_action, daemon=True).start()
            current_action = "Duck"
        else:
            current_action = "Idle"

    return gesture, confidence


# Run Chrome Dinosaur Game
def start_dino_game():
    dino_path = "./dino_main.py"
    subprocess.run(["python", dino_path])

    # Ensure Chrome Dino window get the point
    time.sleep(3)
    pyautogui.click(100, 100)  # Click game window


# Main
def Main():
    global guessGestureMode

    # Start thread to run Chrome Dinosaur
    dino_thread = threading.Thread(target=start_dino_game)
    dino_thread.start()
    time.sleep(2)  # Wait a second

    # Open the camera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        roi = binaryMask(frame, x0, y0, width, height)

        if guessGestureMode:
            label, confidence = guessGesture(model, roi)
            cv2.putText(frame, f"{label} ({confidence:.1f}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Original", frame)
        cv2.imshow("ROI", roi)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # Press ESC to quit
            break
        elif key == ord('g'):
            guessGestureMode = not guessGestureMode
            print(f"Prediction Mode - {guessGestureMode}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Main()