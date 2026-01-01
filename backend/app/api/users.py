from fastapi import APIRouter, Depends, HTTPException
from firebase_admin import auth
from backend.app.core.firebase_config import db
from backend.app.core.auth_utils import verify_firebase_token

router = APIRouter(prefix="/users", tags=["Users"])


# CREATE – SIGNUP
@router.post("/signup")
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


# READ – GET USER INFO
@router.get("/me")
def get_my_profile(user=Depends(verify_firebase_token)):
    doc = db.collection("users").document(user["uid"]).get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    return doc.to_dict()


# UPDATE – FORGOT PASSWORD
@router.post("/forgot-password")
def forgot_password(email: str):
    try:
        link = auth.generate_password_reset_link(email)
        return {
            "message": "Password reset link generated",
            "reset_link": link
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# DELETE – DELETE ACCOUNT
@router.delete("/delete-account")
def delete_account(user=Depends(verify_firebase_token)):
    try:
        auth.delete_user(user["uid"])
        db.collection("users").document(user["uid"]).delete()
        return {"message": "Account deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
