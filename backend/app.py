from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from models import UserCreate, UserLogin
from database import users_collection, records_collection
from auth import get_current_user, hash_password, verify_password, create_access_token
from datetime import datetime
import pytz
import uuid
import os
import base64
import requests

# ---------------------------
# Config
# ---------------------------

HF_API_URL = "https://keshavnayak15-cardiovision-b7.hf.space/run/predict"

IST = pytz.timezone("Asia/Kolkata")

def get_ist_time():
    return datetime.now(IST)

# ---------------------------
# App Setup
# ---------------------------

app = FastAPI(title="CardioVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    img_bytes = await file.read()

    encoded = base64.b64encode(img_bytes).decode("utf-8")

    response = requests.post(
        HF_API_URL,
        json={"data": [f"data:image/png;base64,{encoded}"]}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Model inference failed")

    result = response.json()["data"][0]

    risk = result["risk"]
    confidence = result["confidence"]

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
