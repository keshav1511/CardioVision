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

# ---------------------------
# Config
# ---------------------------

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://keshavnayak15-cardiovision-b7.hf.space"
)

IST = pytz.timezone("Asia/Kolkata")

def get_ist_time():
    return datetime.now(IST)

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
    try:
        # Read uploaded image
        img_bytes = await file.read()

        print("Received file:", file.filename)
        print("File size:", len(img_bytes))
        print("Sending request to HF:", HF_SPACE_URL)

        # Convert image to base64 (required for Gradio)
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Send to Hugging Face Space
        response = requests.post(
            f"{HF_SPACE_URL}/api/predict",
            json={
                "data": [f"data:image/png;base64,{img_base64}"]
            },
            timeout=60
        )

        print("HF STATUS:", response.status_code)
        print("HF RESPONSE:", response.text)

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"HF Error: {response.text}"
            )

        data = response.json()

        # Gradio returns output inside data list
        result = data["data"][0]

        risk = result["risk"]
        confidence = result["confidence"]

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
