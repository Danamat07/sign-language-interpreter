import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

"""
Initializes Firebase Admin SDK once.
Provides Firestore client db to use in APIs.
"""

# load environment variables from .env file
load_dotenv()

# path to firebase admin SDK json key
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH")

if not FIREBASE_CRED_PATH:
    raise RuntimeError("FIREBASE_CRED_PATH is not set")

# initialize firebase admin SDK only if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)

# firestore client for database operations
db = firestore.client()
