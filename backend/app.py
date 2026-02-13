from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import UserCreate, UserLogin
from database import users_collection, records_collection
from auth import get_current_user, hash_password, verify_password, create_access_token
from datetime import datetime
import pytz
import os
from gradio_client import Client
from PIL import Image
import numpy as np
import io

# ---------------------------
# Config
# ---------------------------

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "keshavnayak15/cardiovision-b7"
)

IST = pytz.timezone("Asia/Kolkata")

def get_ist_time():
    return datetime.now(IST)

# Initialize Gradio client
def init_gradio_client():
    try:
        client = Client(HF_SPACE_URL)
        print(f"✅ Connected to Gradio Space: {HF_SPACE_URL}")
        return client
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to Gradio Space: {e}")
        return None

gradio_client = init_gradio_client()

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
# Prediction Route - FIXED
# ---------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    global gradio_client
    
    # Try to reconnect if client is None
    if not gradio_client:
        print("🔄 Attempting to reconnect to Gradio Space...")
        gradio_client = init_gradio_client()
    
    if not gradio_client:
        raise HTTPException(
            status_code=503,
            detail="Prediction service temporarily unavailable. Please try again."
        )
    
    try:
        # Read uploaded image
        img_bytes = await file.read()

        print(f"📤 Received file: {file.filename}")
        print(f"📊 File size: {len(img_bytes)} bytes")

        # Convert to PIL Image then to numpy array
        # This matches the gr.Image(type="numpy") input
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_array = np.array(image)

        print(f"🖼️ Image shape: {image_array.shape}")
        print(f"🚀 Sending to HF Space: {HF_SPACE_URL}")

        # Call Gradio Space with numpy array
        result = gradio_client.predict(
            image_array,  # Send numpy array directly
            api_name="/predict"
        )

        print(f"✅ HF Response: {result}")
        print(f"📥 Response type: {type(result)}")

        # Parse result - your Gradio returns {"risk": str, "confidence": float}
        if isinstance(result, dict):
            risk = result.get("risk", "Unknown")
            confidence = float(result.get("confidence", 0))
        else:
            # Fallback parsing
            risk = str(result)
            confidence = 0.0

        print(f"✅ Parsed - Risk: {risk}, Confidence: {confidence}")

    except Exception as e:
        print(f"❌ PREDICTION ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        gradio_client = None
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

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
    return {
        "status": "CardioVision backend running",
        "gradio_connected": gradio_client is not None
    }