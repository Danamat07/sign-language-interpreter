from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.users import router as users_router

app = FastAPI(
    title="ASL Recognition Backend",
    version="1.0.0",
    description="Backend pentru Firebase Auth + Firestore"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "backend running"}

app.include_router(users_router)
