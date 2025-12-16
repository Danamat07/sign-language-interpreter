"""
Preprocess Kaggle ASL Alphabet dataset using MediaPipe Hands.

This script:
✔ detects the hand
✔ crops based on landmarks
✔ pads to square
✔ places hand on white background
✔ resizes to 128x128
✔ saves to a new dataset folder

IMPORTANT:
- This makes TRAIN data match REAL CAMERA data
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "asl_alphabet",
    "asl_alphabet_train"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "asl_processed"
)

IMG_SIZE = 128

# MediaPipe Setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Helper Function
def process_image(img):
    """Detect hand, crop, pad to square, return processed image"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]

    h, w, _ = img.shape
    xs = [int(lm.x * w) for lm in hand.landmark]
    ys = [int(lm.y * h) for lm in hand.landmark]

    x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
    y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)

    crop = img[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None

    # Make square with white background
    ch, cw, _ = crop.shape
    size = max(ch, cw)
    square = np.ones((size, size, 3), dtype=np.uint8) * 255

    y_off = (size - ch) // 2
    x_off = (size - cw) // 2
    square[y_off:y_off + ch, x_off:x_off + cw] = crop

    square = cv2.resize(square, (IMG_SIZE, IMG_SIZE))
    return square

# Process Dataset
os.makedirs(OUTPUT_DIR, exist_ok=True)

letters = sorted(os.listdir(INPUT_DIR))

print("Processing letters:", letters)

for letter in letters:
    input_letter_dir = os.path.join(INPUT_DIR, letter)
    output_letter_dir = os.path.join(OUTPUT_DIR, letter)
    os.makedirs(output_letter_dir, exist_ok=True)

    images = os.listdir(input_letter_dir)

    for img_name in tqdm(images, desc=f"Processing {letter}"):
        img_path = os.path.join(input_letter_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        processed = process_image(img)

        if processed is not None:
            save_path = os.path.join(output_letter_dir, img_name)
            cv2.imwrite(save_path, processed)

print("\nDataset preprocessing completed.")
print(f"Processed dataset saved to: {OUTPUT_DIR}")
