"""
Real-time ASL Alphabet Recognition (A-Z)
Using CNN + MediaPipe Hands
Enhanced for stability with padded hand crop and prediction history.
"""

import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
from collections import Counter

# Load Model & Classes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/cnn_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "../models/classes.json")

model = load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    CLASSES = json.load(f)

IMG_SIZE = model.input_shape[1:3]  # (128, 128)

print("Loaded classes:", CLASSES)

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Utilities
def preprocess(img):
    """Resize, normalize and expand dims for CNN input"""
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_letter(img):
    """Predict ASL letter from a square image"""
    preds = model.predict(preprocess(img), verbose=0)[0]
    idx = np.argmax(preds)
    return CLASSES[idx], preds[idx]

# Main Loop
def main():
    cap = cv2.VideoCapture(0)
    history = []  # keep last few predictions for stability

    print("Camera started (press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        letter = ""
        conf = 0.0

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Bounding box from landmarks
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in hand.landmark]
            ys = [int(lm.y * h) for lm in hand.landmark]

            x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
            y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)

            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size > 0:
                # Make square with white background
                hand_h, hand_w, _ = hand_crop.shape
                size = max(hand_h, hand_w)
                square = np.ones((size, size, 3), dtype=np.uint8) * 255

                y_off = (size - hand_h) // 2
                x_off = (size - hand_w) // 2

                square[y_off:y_off + hand_h, x_off:x_off + hand_w] = hand_crop

                # Predict letter
                letter, conf = predict_letter(square)

                # Add to history
                history.append(letter)
                if len(history) > 7:
                    history.pop(0)

                # Stabilize J and Z
                if letter in ["J", "Z"]:
                    letter = Counter(history).most_common(1)[0][0]

                # Draw rectangle
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Display result
        display_text = f"Letter: {letter} ({conf*100:.1f}%)" if letter else "Show ASL letter"
        cv2.putText(frame, display_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sign Language Interpreter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
