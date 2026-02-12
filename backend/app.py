from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2
import os
from fastapi.staticfiles import StaticFiles
from models import UserCreate, UserLogin
from database import users_collection, records_collection
from auth import get_current_user, hash_password, verify_password, create_access_token
from datetime import datetime
import pytz
import uuid
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests


# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

app.mount("/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")

# ---------------------------
# Model Download + Loading
# ---------------------------

MODEL_PATH = os.path.join(BASE_DIR, "cardiovision_b7.pth")
MODEL_URL = "https://github.com/keshav1511/CardioVision/releases/download/v1.0/cardiovision_b7.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from GitHub Release...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("✅ Model downloaded successfully")
        else:
            raise RuntimeError("❌ Failed to download model")

download_model()

model = EfficientNet.from_name("efficientnet-b7")
model._fc = torch.nn.Linear(model._fc.in_features, 1)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

print("✅ EfficientNet-B7 loaded successfully!")


# ---------------------------
# Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def risk_level(p):
    if p < 0.25:
        return "Low Risk"
    elif p < 0.60:
        return "Moderate Risk"
    return "High Risk"

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    img_bytes = await file.read()

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    risk = risk_level(prob)

    filename = f"{uuid.uuid4()}.png"
    file_path = os.path.join(HEATMAP_DIR, filename)

    cv2.imwrite(file_path, cv2.resize(np.array(image), (300, 300)))

    host = os.getenv("API_HOST", "")
    heatmap_url = f"{host}/heatmaps/{filename}" if host else f"/heatmaps/{filename}"

    records_collection.insert_one({
        "user_id": user_id,
        "risk": risk,
        "confidence": round(prob * 100, 2),
        "heatmap_path": heatmap_url,
        "created_at": get_ist_time()
    })

    return {
        "risk": risk,
        "confidence": round(prob * 100, 2),
        "heatmap_url": heatmap_url
    }

# ---------------------------
# Health
# ---------------------------
@app.get("/")
def root():
    return {
        "status": "CardioVision API running",
        "model_loaded": model is not None,
        "device": str(device)
    }
