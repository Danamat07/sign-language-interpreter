from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth

"""
Verifies Firebase JWT tokens sent from frontend.
Used as a dependency in endpoints to authenticate users.
"""

# HTTPBearer security scheme for fastapi
security = HTTPBearer()

def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Verify Firebase ID token sent in Authorization header.

    Parameters:
        credentials: HTTPAuthorizationCredentials injected by FastAPI

    Returns:
        dict: Decoded token containing user info

    Raises:
        HTTPException: 401 if token is invalid or expired
    """
    token = credentials.credentials
    try:
        # decode firebase ID token
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
