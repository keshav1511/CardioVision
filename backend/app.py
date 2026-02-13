from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import UserCreate, UserLogin
from database import users_collection, records_collection
from auth import get_current_user, hash_password, verify_password, create_access_token
from datetime import datetime
import pytz
import os
import requests
import base64
from gradio_client import Client

# ---------------------------
# Config
# ---------------------------

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "keshavnayak15-cardiovision-b7.hf.space"
)

IST = pytz.timezone("Asia/Kolkata")

def get_ist_time():
    return datetime.now(IST)

try:
    gradio_client = Client(HF_SPACE_URL)
    print(f"✅ Connected to Gradio Space: {HF_SPACE_URL}")
except Exception as e:
    print(f"⚠️ Warning: Could not connect to Gradio Space: {e}")
    gradio_client = None

# ---------------------------
# App Setup
# ---------------------------

app = FastAPI(title="CardioVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Auth Routes
# ---------------------------

@app.post("/signup")
def signup(user: UserCreate):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    users_collection.insert_one({
        "username": user.username,
        "email": user.email,
        "password": hash_password(user.password),
        "created_at": get_ist_time()
    })

    return {"message": "Account created successfully"}


@app.post("/login")
def login(user: UserLogin):
    db_user = users_collection.find_one({"email": user.email})

    if not db_user:
        raise HTTPException(status_code=404, detail="Email not found")

    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect password")

    access_token = create_access_token(
        data={"sub": str(db_user["_id"]), "email": db_user["email"]}
    )

    return {"access_token": access_token, "token_type": "bearer"}


# ---------------------------
# Prediction Route
# ---------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    if not gradio_client:
        raise HTTPException(
            status_code=503,
            detail="Prediction service unavailable"
        )
    
    try:
        # Read uploaded image
        img_bytes = await file.read()

        print("Received file:", file.filename)
        print("File size:", len(img_bytes))

        # ✅ Save temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        # ✅ Call Gradio Space
        result = gradio_client.predict(
            image=temp_path,
            api_name="/predict"  # Check your Gradio interface for correct name
        )

        print("HF RESPONSE:", result)

        # Parse result based on your Gradio output format
        if isinstance(result, dict):
            risk = result.get("risk", "Unknown")
            confidence = result.get("confidence", 0)
        else:
            risk = str(result)
            confidence = 0

        # Clean up temp file
        os.remove(temp_path)

    except Exception as e:
        print("ERROR OCCURRED:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # Save prediction record
    records_collection.insert_one({
        "user_id": user_id,
        "risk": risk,
        "confidence": confidence,
        "created_at": get_ist_time()
    })

    return {
        "risk": risk,
        "confidence": confidence
    }

# ---------------------------
# Health Check
# ---------------------------

@app.get("/")
def root():
    return {"status": "CardioVision backend running"}
