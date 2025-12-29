import numpy as np
import tensorflow as tf
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "cnn_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "classes.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH) as f:
    CLASSES = json.load(f)

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.7

def predict_hand(image):
    prediction = model.predict(image, verbose=0)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence

    return CLASSES[class_index], confidence
