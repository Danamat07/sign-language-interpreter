import os
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

"""
Initializes Firebase Admin SDK once.
Provides Firestore client db to use in APIs.
"""

load_dotenv()

firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")

if not firebase_credentials:
    raise RuntimeError("FIREBASE_CREDENTIALS is not set")

cred_dict = json.loads(firebase_credentials)

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()