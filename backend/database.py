from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MONGO_URL = os.getenv("MONGODB_URL")

if not MONGO_URL:
    raise RuntimeError(
        "❌ MONGODB_URL not found. Please set it in .env file."
    )


try:
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    # Test connection
    client.server_info()
    print("✅ Successfully connected to MongoDB")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {e}")
    raise

db = client["cardiovision_db"]

users_collection = db["users"]
records_collection = db["records"]

# Index Creations
try:
    users_collection.create_index("email", unique=True)
    records_collection.create_index("user_id")
    records_collection.create_index("created_at")
    print("✅ Database indexes created")
except Exception as e:
    print(f"⚠️ Index creation warning: {e}")