from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
import cv2
import numpy as np

from src.auth_utils import verify_firebase_token
from src.ml.asl_model import predict_hand
from src.ml.preprocess import preprocess_hand_image

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("")
async def predict(
    file: UploadFile = File(...),
    user=Depends(verify_firebase_token)
):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    processed = preprocess_hand_image(img)
    letter, confidence = predict_hand(processed)

    return {
        "letter": letter,
        "confidence": float(confidence)
    }
