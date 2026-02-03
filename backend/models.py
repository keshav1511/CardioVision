from pydantic import BaseModel, EmailStr, Field
from datetime import datetime


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=64)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class PredictionResponse(BaseModel):
    risk: str
    confidence: float
    heatmap_url: str


class RecordResponse(BaseModel):
    user_id: str
    risk: str
    confidence: float
    heatmap_path: str
    created_at: datetime