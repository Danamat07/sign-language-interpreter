from firebase_config import db
from firebase_admin import auth

def create_user(email, password, username):
    try:
        # creaza user in firebase auth
        user = auth.create_user(
            email = email,
            password = password,
            display_name = username
        )
        # creaza document firestore pentru info suplimentare
        db.collection("users").document(user.uid).set({
            "username": username,
            "email": email
        })
        return user.uid
    except Exception as e:
        print("Error creating user:", e)
        return None
    
def update_password(uid, new_password):
    try:
        auth.update_user(uid, password=new_password)
        return True
    except Exception as e:
        print("Error updating password:", e)
        return False
    
def delete_user(uid):
    try:
        auth.delete_user(uid)
        db.collection("users").document(uid).delete()
        return True
    except Exception as e:
        print("Error deleting user:", e)
        return False
    
def log_gesture(uid, gesture, timestamp):
    try:
        db.collection("users").document(uid).collection("gestures").add({
            "gesture": gesture,
            "timestamp": timestamp
        })
        return True
    except Exception as e:
        print("Error logging gesture:", e)
        return False