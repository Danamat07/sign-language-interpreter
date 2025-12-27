import firebase_admin
from firebase_admin import credentials, auth, firestore

# initializare firebase admin sdk folosind cheia de serviciu
cred = credentials.Certificate("src/config/firebase_key.json")
firebase_admin.initialize_app(cred)

# firestore client
db = firestore.client()
