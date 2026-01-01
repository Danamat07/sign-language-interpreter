from fastapi import APIRouter, Depends, HTTPException
from firebase_admin import auth
from backend.app.core.firebase_config import db
from backend.app.core.auth_utils import verify_firebase_token

"""
API endpoints for user management (CRUD).
"""

# create fastapi router for user-related endpoints
router = APIRouter(prefix="/users", tags=["Users"])


# CREATE – SIGNUP
@router.post("/signup")
def signup(email: str, password: str, username: str):
    """
    Create a new user in Firebase Auth and store user info in Firestore.

    Parameters:
        email (str): User email
        password (str): User password
        username (str): Display name / username

    Returns:
        dict: Success message
    """
    try:
        # create user in firebase authentication
        user = auth.create_user(
            email=email,
            password=password,
            display_name=username
        )

        # store additional use info in firestore unser "users" collection
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
    """
    Retrieve the currently logged-in user's profile from Firestore.

    Parameters:
        user: Decoded Firebase token (injected by dependency)

    Returns:
        dict: User profile
    """
    doc = db.collection("users").document(user["uid"]).get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    return doc.to_dict()


# UPDATE – FORGOT PASSWORD
@router.post("/forgot-password")
def forgot_password(email: str):
    """
    Generate a password reset link for the given email.

    Parameters:
        email (str): User email

    Returns:
        dict: Success message and password reset link
    """
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
    """
    Delete a user account from Firebase Auth and Firestore.

    Parameters:
        user: Decoded Firebase token (injected by dependency)

    Returns:
        dict: Success message
    """
    try:
        # delete user from firebase auth
        auth.delete_user(user["uid"])
        # delete firestore document
        db.collection("users").document(user["uid"]).delete()
        return {"message": "Account deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
