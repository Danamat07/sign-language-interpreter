import cv2
from collections import deque
from backend.app.ml.asl_model import predict_letter
from backend.app.ml.preprocess import preprocess_hand_image
import mediapipe as mp

"""
Hand detection and cropping logic using MediaPipe.
This extracts hand bounding boxes to feed into CNN.
"""

# majority voting
VOTING_FRAMES = 7
CONFIDENCE_THRESHOLD = 0.7
prediction_buffer = deque(maxlen=VOTING_FRAMES)

# mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils


def detect_and_predict(frame, draw=True):
    """
    Detects a hand, predicts ASL letter using CNN, applies majority voting, 
    and optionally draws bounding box and landmarks.

    Returns:
        predicted_letter (str | None)
        confidence (float)
        annotated_frame (np.ndarray)
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    predicted_letter = None
    final_confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            # compute bounding box
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.size == 0:
                continue

            # preprocess + CNN prediction
            input_tensor = preprocess_hand_image(hand_crop)
            letter, confidence = predict_letter(input_tensor)

            final_confidence = confidence

            # push into voting buffer only if confident
            if letter and confidence >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(letter)

            # majority voting
            if len(prediction_buffer) == prediction_buffer.maxlen:
                predicted_letter = max(
                    set(prediction_buffer),
                    key=prediction_buffer.count
                )
            else:
                predicted_letter = letter

            # draw overlays
            if draw:
                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

    return predicted_letter, final_confidence, frame