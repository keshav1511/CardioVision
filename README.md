# CardioVision - AI-Based Heart Disease Detection

CardioVision is a web application that uses deep learning to detect heart disease risk from retinal fundus images.

## Features

- 🔐 User authentication (signup/login)
- 🤖 AI-powered heart disease risk prediction
- 🔥 Grad-CAM heatmap visualization
- 📄 PDF medical report generation
- 💾 MongoDB database for user records

## Technology Stack

**Backend:**
- FastAPI
- PyTorch (EfficientNet-B7)
- MongoDB
- JWT Authentication

**Frontend:**
- Vanilla JavaScript
- HTML5/CSS3

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- MongoDB account (or local MongoDB)
- Your trained model file: `cardiovision_b7.pth`

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd cardiovision

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
MONGODB_URL=your_mongodb_connection_string
SECRET_KEY=your_secret_key_change_this_in_production
API_HOST=http://127.0.0.1:8000
ENV=development
```

### 4. Add Model File

Place your trained model file `cardiovision_b7.pth` in the project root directory.

### 5. Run the Application

```bash
# Start the FastAPI server
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 6. Access the Application

Open your browser and navigate to:
- Frontend: Open `login.html` in your browser
- API Docs: http://127.0.0.1:8000/docs

## Project Structure

```
CardioVision/
├── backend/
│ ├── app.py                        # FastAPI application
│ ├── auth.py                       # JWT authentication logic
│ ├── database.py                   # MongoDB connection
│ ├── models.py                     # Pydantic models
│ ├── cardiovision_b7.pth           # Trained model (NOT committed)
│ └── heatmaps/                     # Generated Grad-CAM images
│
├── frontend/
│ ├── login.html                    # Login page
│ ├── signup.html                   # Signup page
│ ├── dashboard.html                #Main Dashboard
│ ├── app.js                        #Frontend JavaScript
│ └── style.css                     #Styling
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## API Endpoints

### Authentication
- `POST /signup` - Create new user account
- `POST /login` - Login and get JWT token

### Prediction
- `POST /predict` - Upload image and get prediction (requires auth)
- `GET /download-report` - Download PDF report (requires auth)

### Health Check
- `GET /` - API information
- `GET /health` - Health check

## Security Features

✅ Password hashing with bcrypt
✅ JWT token authentication
✅ File size validation (10MB max)
✅ File type validation
✅ Environment variable configuration
✅ CORS protection

## Error Fixes Applied

1. ✅ Fixed import error (`database` → `databases`)
2. ✅ Fixed PDF image path issue
3. ✅ Added environment variables for secrets
4. ✅ Added file validation (size & type)
5. ✅ Added model file existence check
6. ✅ Improved error handling
7. ✅ Added loading states in UI
8. ✅ Made API URL configurable

## Usage

1. **Sign Up**: Create a new account
2. **Login**: Login with your credentials
3. **Upload**: Upload a retinal fundus image
4. **Analyze**: Click "Analyze Image" to get prediction
5. **Download**: Download the medical report as PDF

## Notes

- Ensure MongoDB is running and accessible
- Model file must be present before starting the server
- For production, change SECRET_KEY and use HTTPS
- Adjust CORS settings for production deployment

## Troubleshooting

**Issue**: Model file not found
- **Solution**: Ensure `cardiovision_b7.pth` is in the project root

**Issue**: MongoDB connection failed
- **Solution**: Check your `MONGODB_URL` in `.env` file

**Issue**: Cannot access API from frontend
- **Solution**: Ensure CORS is properly configured and server is running

## License

MIT License

## Contact

For issues or questions, please open an issue in the repository.