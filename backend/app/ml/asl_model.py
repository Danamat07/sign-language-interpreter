import tensorflow as tf
import numpy as np
import json
from pathlib import Path

"""
Loads the trained CNN model for ASL recognition and performs predictions.
This is the core ML logic for inference.
"""

# paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "cnn_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "classes.json"

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# load class labels
with open(CLASSES_PATH, "r") as f:
    CLASSES = json.load(f)

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.7

def predict_letter(image_tensor):
    """
    Predict the letter from a preprocessed image tensor.
    Returns (letter, confidence) or (None, confidence) if below threshold.
    """
    prediction = model.predict(image_tensor, verbose=0)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index])

    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence

    return CLASSES[class_index], confidence
