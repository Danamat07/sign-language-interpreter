# src/main.py

from fastapi import FastAPI, Depends, HTTPException
from firebase_admin import auth
from src.firebase_config import db
from src.auth_utils import verify_firebase_token
from src.api.predict import router as predict_router

app = FastAPI(
    title="ASL Recognition Backend",
    version="0.1.0",
    description="Backend pentru recunoașterea alfabetului ASL folosind CNN"
)

app.include_router(predict_router)


# SIGNUP

@app.post("/signup")
def signup(email: str, password: str, username: str):
    try:
        user = auth.create_user(
            email=email,
            password=password,
            display_name=username
        )
        db.collection("users").document(user.uid).set({
            "email": email,
            "username": username
        })
        return {"message": "User created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# LOGIN

"""
Login-ul NU se face în backend!
Frontend-ul folosește Firebase Auth SDK.
Backend-ul doar validează token-ul.
"""


# UPDATE PASSWORD

@app.post("/update-password")
def update_password(
    new_password: str,
    user=Depends(verify_firebase_token)
):
    try:
        auth.update_user(user["uid"], password=new_password)
        return {"message": "Password updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# FORGOT PASSWORD

@app.post("/forgot-password")
def forgot_password(email: str):
    try:
        link = auth.generate_password_reset_link(email)
        return {
            "message": "Password reset link generated",
            "reset_link": link
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# DELETE ACCOUNT

@app.delete("/delete-account")
def delete_account(user=Depends(verify_firebase_token)):
    try:
        auth.delete_user(user["uid"])
        db.collection("users").document(user["uid"]).delete()
        return {"message": "Account deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
