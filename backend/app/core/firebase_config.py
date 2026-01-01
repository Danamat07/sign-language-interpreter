import firebase_admin
from firebase_admin import credentials, firestore

# path catre cheia firebase admin
FIREBASE_CRED_PATH = "backend/app/config/asl-recognition-8a208-firebase-adminsdk-fbsvc-597c771d23.json"

# initializare firebase (o singura data)
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)

# firestore client
db = firestore.client()
