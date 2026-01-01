from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.users import router as users_router

"""
Sets up FastAPI application.
Adds CORS so frontend can communicate with backend.
Includes all /users endpoints.
"""

# initialize fastapi app
app = FastAPI(
    title="ASL Recognition Backend",
    version="1.0.0",
    description="Backend for Firebase Auth + Firestore"
)

# enable CORS (cross-origin resource sharing).
# this allows frontend running on another origin to make requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins (development only)
    allow_credentials=True,
    allow_methods=["*"],        # allow GET, POST, DELETE, etc.
    allow_headers=["*"],        # allow headers like authorization
)

# root endpoint to check if backend is running
@app.get("/")
def root():
    return {"status": "backend running"}

# include the users router for all /users endpoints
app.include_router(users_router)
