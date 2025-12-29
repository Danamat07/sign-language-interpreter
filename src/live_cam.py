import cv2   
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
from collections import deque
from pathlib import Path

# director root
BASE_DIR = Path(__file__).resolve().parent.parent
# path model cnn
MODEL_PATH = BASE_DIR / "models" / "cnn_model.keras"
# path clase
CLASSES_PATH = BASE_DIR / "models" / "classes.json"

# mdelul antrenat
model = tf.keras.models.load_model(MODEL_PATH)

# lista de clase
with open(CLASSES_PATH, "r") as f:
    CLASSES = json.load(f)

print("Classes:", CLASSES)

IMG_SIZE = 128                  # dimensiune imagine
CONFIDENCE_THRESHOLD = 0.7      # prag min incredere
VOTING_FRAMES = 7               # frame-uri majority voting

# buffer pt ultimele predictii
prediction_buffer = deque(maxlen=VOTING_FRAMES)

# media pipe - hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode = False,      # video stream
    max_num_hands = 1,              # o singura mana
    min_detection_confidence = 0.7, # cat de sigur sa fie detectorul
    min_tracking_confidence = 0.7   # stabilitate tracking
)

mp_draw = mp.solutions.drawing_utils

# functie preprocesare
def preprocess_hand_image(hand_img):
    """
    preprocesare identica cu training-ul:
        - resize img la 128x128
        - conversie din BGR (opencv) in RGB
        - normalizare valori pixeli
    """
    hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    hand_img = hand_img / 255.0
    hand_img = np.expand_dims(hand_img, axis=0)
    return hand_img

# pornire camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()
print("Camera started. Press 'q' to quit.")

# loop principal
while True:

    # citim un frame din camera
    ret, frame = cap.read()
    if not ret:
        break

    # oglindim imaginea
    frame = cv2.flip(frame, 1)

    # copie pt afisare
    display_frame = frame.copy()

    # convertim frame-ul in RGB pt mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detectam mana
    result = hands.process(rgb_frame)

    predicted_letter = ""

    # daca a fost detectata o mana
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # extragem coordonatele landmark-urilor
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            # bounding box in jurul mainii
            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

            # crop pe mana
            hand_crop = frame[y_min:y_max, x_min:x_max]

            # verificam daca imaginea e valida
            if hand_crop.size == 0:
                continue

            # preprocesare pt cnn
            input_tensor = preprocess_hand_image(hand_crop)

            # predictie cnn
            predinctions = model.predict(input_tensor, verbose=0)[0]

            # clasa pt probabilitatea max
            class_index = np.argmax(predinctions)
            confidence = predinctions[class_index]

            # daca predictia este suficient de sigura
            if confidence > CONFIDENCE_THRESHOLD:
                prediction_buffer.append(class_index)

            # stabilizare - majority voting
            if len(prediction_buffer) == VOTING_FRAMES:
                stable_class = max(set(prediction_buffer), key=prediction_buffer.count)
                predicted_letter = CLASSES[stable_class]

            # desenam bounding box-ul
            cv2.rectangle(
                display_frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            # desenam landmark-urile
            mp_draw.draw_landmarks(
                display_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
        
    # afisam litera detectata
    cv2.putText(
        display_frame,
        f"Letter: {predicted_letter}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # afisam frame-ul
    cv2.imshow("ASL Interpreter", display_frame)

    # iesire cu tasta q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
