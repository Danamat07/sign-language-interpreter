import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
from collections import deque

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.7
VOTING_FRAMES = 7

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "cnn_model.keras")
CLASSES_PATH = os.path.join(PROJECT_DIR, "models", "classes.json")

# ==============================
# LOAD MODEL & CLASSES
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

# ==============================
# MEDIAPIPE HANDS
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==============================
# PREDICTION STABILIZATION
# ==============================
prediction_buffer = deque(maxlen=VOTING_FRAMES)

def majority_vote(buffer):
    if not buffer:
        return None
    return max(set(buffer), key=buffer.count)

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

print("📷 Camera started. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_letter = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # ==============================
            # HAND BOUNDING BOX
            # ==============================
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w) - 20
            x_max = int(max(x_coords) * w) + 20
            y_min = int(min(y_coords) * h) - 20
            y_max = int(max(y_coords) * h) + 20

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                # ==============================
                # PREPROCESSING (IDENTIC CU TRAINING)
                # ==============================
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                # ==============================
                # PREDICTION
                # ==============================
                preds = model.predict(hand_img, verbose=0)[0]
                confidence = np.max(preds)
                class_idx = np.argmax(preds)

                if confidence > CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(classes[class_idx])
                else:
                    prediction_buffer.append("")

                voted = majority_vote(prediction_buffer)
                predicted_letter = voted if voted else ""

            # DRAW BOX
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ==============================
    # DISPLAY
    # ==============================
    cv2.putText(
        frame,
        f"Letter: {predicted_letter}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        3
    )

    cv2.imshow("ASL Interpreter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
