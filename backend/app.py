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

# Load environment variables
load_dotenv()

# Set Indian timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in Indian Standard Time"""
    return datetime.now(IST)

# ----------------------------------
# App
# ----------------------------------
app = FastAPI(title="CardioVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cardio-vision-nine.vercel.app",
        "https://cardio-vision-opv3u9519-keshav-nayaks-projects.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Absolute path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

app.mount("/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Auth Routes
@app.post("/signup")
def signup(user: UserCreate):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    users_collection.insert_one({
        "username": user.username,
        "email": user.email,
        "password": hash_password(user.password),
        "created_at": get_ist_time()  # ✅ FIXED: IST time
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


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if model file exists
model_path = "cardiovision_b7.pth"
if not os.path.exists(model_path):
    print(f"⚠️ Warning: Model file '{model_path}' not found!")
    model = None
else:
    model = EfficientNet.from_name("efficientnet-b7")
    model._fc = torch.nn.Linear(model._fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = self.model._blocks[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        output = self.model(input_tensor)

        self.model.zero_grad()
        output.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (300, 300))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


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


# Prediction
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please contact administrator.")
    
    try:
        # Validate file size
        img_bytes = await file.read()
        
        if len(img_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        
        # Load and process image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        original = cv2.resize(np.array(image), (300, 300))
        tensor = transform(image).unsqueeze(0).to(device)
        tensor.requires_grad = True

        # Get prediction
        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()
        
        risk = risk_level(prob)

        # Generate heatmap
        tensor.requires_grad = True
        cam = GradCAM(model).generate(tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        # Save heatmap
        filename = f"{uuid.uuid4()}.png"
        file_path = os.path.join(HEATMAP_DIR, filename)
        cv2.imwrite(file_path, overlay)

        # Use environment variable or request info for URL
        host = os.getenv("API_HOST", "http://127.0.0.1:8000")
        heatmap_url = f"{host}/heatmaps/{filename}"

        # Save with IST time
        current_time = get_ist_time()
        
        records_collection.insert_one({
            "user_id": user_id,
            "risk": risk,
            "confidence": round(prob * 100, 2),
            "heatmap_path": heatmap_url,
            "heatmap_filename": filename,
            "created_at": current_time 
        })

        print(f"✅ Prediction successful - Risk: {risk}, Confidence: {prob*100:.2f}%")
        print(f"📸 Heatmap saved: {filename}")
        print(f"⏰ Time: {current_time.strftime('%Y-%m-%d %I:%M:%S %p IST')}")

        return {
            "risk": risk,
            "confidence": round(prob * 100, 2),
            "heatmap_url": heatmap_url,
            "timestamp": current_time.strftime('%Y-%m-%d %I:%M:%S %p IST') 
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Report
@app.get("/download-report")
def download_report(user_id: str = Depends(get_current_user)):
    try:
        record = records_collection.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)]
        )

        if not record:
            raise HTTPException(status_code=404, detail="No record found")

        # Generate unique filename for report
        report_filename = f"CardioVision_Report_{uuid.uuid4()}.pdf"
        report_path = os.path.join(REPORTS_DIR, report_filename)

        # Create PDF
        c = canvas.Canvas(report_path, pagesize=A4)
        width, height = A4

        #  Get current time in IST
        current_time = get_ist_time()
        
        # Header
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 50, "CardioVision Medical Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Generated on: {current_time.strftime('%d %B %Y, %I:%M:%S %p IST')}")

        # Patient Information
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 120, "Analysis Results:")
        
        c.setFont("Helvetica", 14)
        c.drawString(50, height - 150, f"Risk Level: {record['risk']}")
        c.drawString(50, height - 175, f"Confidence: {record['confidence']}%")
        
        # Display analysis date in IST
        analysis_time = record['created_at']
        if isinstance(analysis_time, datetime):
            # If already timezone-aware, convert to IST
            if analysis_time.tzinfo is not None:
                analysis_time = analysis_time.astimezone(IST)
            else:
                # If naive datetime (UTC), localize then convert
                analysis_time = pytz.utc.localize(analysis_time).astimezone(IST)
            
            formatted_time = analysis_time.strftime('%d %B %Y, %I:%M:%S %p IST')
        else:
            formatted_time = str(analysis_time)
        
        c.drawString(50, height - 200, f"Analysis Date: {formatted_time}")

        # Use local file path for image
        if 'heatmap_filename' in record:
            img_path = os.path.join(HEATMAP_DIR, record['heatmap_filename'])
        else:
            filename = record["heatmap_path"].split("/")[-1]
            img_path = os.path.join(HEATMAP_DIR, filename)

        if os.path.exists(img_path):
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 240, "Heatmap Visualization:")
            c.drawImage(img_path, 50, height - 570, width=300, height=300)
        else:
            c.drawString(50, height - 240, "Heatmap image not available")

        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 50, "This report is generated by CardioVision AI system.")
        c.drawString(50, 35, "Please consult with a healthcare professional for medical advice.")

        c.save()

        return FileResponse(
            report_path, 
            filename=report_filename,
            media_type='application/pdf'
        )
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# Health Check
@app.get("/")
def root():
    current_time = get_ist_time()
    return {
        "message": "CardioVision API is running",
        "version": "1.0.0",
        "server_time": current_time.strftime('%d %B %Y, %I:%M:%S %p IST'),  
        "timezone": "Asia/Kolkata (IST)",
        "endpoints": {
            "signup": "/signup",
            "login": "/login",
            "predict": "/predict",
            "download_report": "/download-report"
        }
    }


@app.get("/health")
def health_check():
    current_time = get_ist_time()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "server_time": current_time.strftime('%d %B %Y, %I:%M:%S %p IST'),  
        "timezone": "Asia/Kolkata (IST)"
    }