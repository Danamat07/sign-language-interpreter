import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH")

if not FIREBASE_CRED_PATH:
    raise RuntimeError("FIREBASE_CRED_PATH is not set")

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()
