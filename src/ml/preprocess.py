import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_hand_image(hand_img):
    hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    hand_img = hand_img / 255.0
    hand_img = np.expand_dims(hand_img, axis=0)
    return hand_img
