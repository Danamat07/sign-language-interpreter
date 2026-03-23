from fastapi import APIRouter, Depends, HTTPException
from firebase_admin import auth, firestore
from backend.app.core.firebase_config import db
from backend.app.core.auth_utils import verify_firebase_token

"""
API endpoints for user management (CRUD) + ASL statistics.
"""

# create fastapi router for user-related endpoints
router = APIRouter(prefix="/users", tags=["Users"])


# alphabet WITHOUT J and Z
ALPHABET = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I",
    "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y"
]


# CREATE – SIGNUP
@router.post("/signup")
def signup(email: str, password: str, username: str):

    try:

        # create firebase auth user
        user = auth.create_user(
            email=email,
            password=password,
            display_name=username
        )

        # initialize progress
        progress = {
            letter: False for letter in ALPHABET
        }

        # save firestore user
        db.collection("users").document(user.uid).set({

            "email": email,
            "username": username,
            "progress": progress

        })

        return {
            "message": "User created successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# READ – GET USER INFO
@router.get("/me")
def get_my_profile(user=Depends(verify_firebase_token)):

    doc = db.collection("users").document(user["uid"]).get()

    if not doc.exists:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

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
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# SAVE PROGRESS
@router.post("/recognize-letter")
def recognize_letter(data: dict, user=Depends(verify_firebase_token)):

    letter = data.get("letter")

    if letter not in ALPHABET:
        raise HTTPException(status_code=400, detail="Invalid letter")

    user_ref = db.collection("users").document(user["uid"])
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()

    recognized = user_data.get("recognizedLetters", {})

    # dacă deja e true
    if recognized.get(letter):
        return {"message": "Letter already learned"}

    # update direct în recognizedLetters
    user_ref.update({
        f"recognizedLetters.{letter}": True
    })

    # history (opțional, dar util)
    user_ref.collection("history").add({
        "letter": letter,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    return {"message": "Letter marked as learned"}


# DELETE – DELETE ACCOUNT
@router.delete("/delete-account")
def delete_account(user=Depends(verify_firebase_token)):

    try:

        uid = user["uid"]
        user_ref = db.collection("users").document(uid)
        user_doc = user_ref.get()

        if user_doc.exists:

            # delete history subcollection
            history_docs = user_ref.collection("history").stream()

            for doc in history_docs:
                doc.reference.delete()

            # delete firestore user document
            user_ref.delete()

        # delete firebase auth user
        auth.delete_user(uid)

        return {
            "message": "Account deleted successfully"
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# SAVE GAME SCORE
@router.post("/save-score")
def save_score(data: dict, user=Depends(verify_firebase_token)):

    score = data.get("score", 0)

    user_ref = db.collection("users").document(user["uid"])
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()

    current_highscore = user_data.get("highscore", 0)

    # update only if better
    if score > current_highscore:
        user_ref.update({
            "highscore": score
        })

    return {
        "message": "Score processed"
    }