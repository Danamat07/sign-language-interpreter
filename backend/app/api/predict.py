from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
import cv2
import numpy as np
import base64
from backend.app.ml.hand_detector import detect_and_predict
from backend.app.core.auth_utils import verify_firebase_token

"""
FastAPI router for ASL predictions.
Receives image, returns predicted letter, confidence, and annotated image.
"""

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("")
async def predict(
    file: UploadFile = File(...), 
    user=Depends(verify_firebase_token)
):
    """
    Accepts an uploaded image, detects hand, predicts ASL letter,
    applies majority voting, and returns annotated frame.
    """
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    letter, confidence, annotated_frame = detect_and_predict(
        frame,
        draw=True
    )

    _, buffer = cv2.imencode(".jpg", annotated_frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "letter": letter or "",
        "confidence": float(confidence),
        "image": img_base64
    }