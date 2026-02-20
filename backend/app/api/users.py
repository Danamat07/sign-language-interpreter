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

        # initialize recognized letters
        recognized_letters = {
            letter: False for letter in ALPHABET
        }

        # save firestore user
        db.collection("users").document(user.uid).set({

            "email": email,
            "username": username,
            "recognizedLetters": recognized_letters

        })

        # increment total users count
        db.collection("metadata").document("usersCount").set({
            "total": firestore.Increment(1)
        }, merge=True)

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


# SAVE RECOGNIZED LETTER
@router.post("/recognize-letter")
def recognize_letter(data: dict, user=Depends(verify_firebase_token)):

    letter = data.get("letter")

    if letter not in ALPHABET:
        raise HTTPException(
            status_code=400,
            detail="Invalid letter"
        )

    user_ref = db.collection("users").document(user["uid"])
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    user_data = user_doc.to_dict()

    # prevent duplicate counting
    if user_data["recognizedLetters"].get(letter):
        return {
            "message": "Letter already saved"
        }

    # update user progress
    user_ref.update({
        f"recognizedLetters.{letter}": True
    })

    # save history for user's personal tracking
    user_ref.collection("history").add({
        "letter": letter,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    # update global statistics – simple count only
    db.collection("statistics").document("global").set({
        letter: firestore.Increment(1)
    }, merge=True)

    return {
        "message": "Letter saved successfully"
    }


# GET GLOBAL STATISTICS
@router.get("/statistics")
def get_statistics(user=Depends(verify_firebase_token)):

    user_doc = db.collection("users").document(user["uid"]).get()
    stats_doc = db.collection("statistics").document("global").get()
    metadata_doc = db.collection("metadata").document("usersCount").get()

    if not user_doc.exists:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return {
        "userProgress": user_doc.to_dict()["recognizedLetters"],
        "globalStats": stats_doc.to_dict(),
        "totalUsers": metadata_doc.to_dict()["total"]
    }


# DELETE – DELETE ACCOUNT
@router.delete("/delete-account")
def delete_account(user=Depends(verify_firebase_token)):

    try:
        # delete firebase auth
        auth.delete_user(user["uid"])

        # delete firestore user
        db.collection("users").document(user["uid"]).delete()

        # decrement total users count
        db.collection("metadata").document("usersCount").set({
            "total": firestore.Increment(-1)
        }, merge=True)

        return {
            "message": "Account deleted"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
