import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_key"),
    api_secret=os.getenv("API_SECRET"),
    secure=True
)

def upload(path):
    result = cloudinary.uploader.upload_large(
        path,
        resource_type='video',
        folder='deepfake-outputs'
    )
    print(result['secure_url'])

    return result['secure_url']