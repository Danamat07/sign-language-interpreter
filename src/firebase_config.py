import firebase_admin
from firebase_admin import credentials, auth, firestore

# initializare firebase admin sdk folosind cheia de serviciu
cred = credentials.Certificate("src/config/asl-recognition-8a208-firebase-adminsdk-fbsvc-597c771d23.json")
firebase_admin.initialize_app(cred)

# firestore client
db = firestore.client()
