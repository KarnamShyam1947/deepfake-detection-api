import os
import io
import cv2
import tempfile
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

def upload_from_memory(video_frames, fps=30.0, filename=None):
    """
    Upload video directly to Cloudinary from memory using frames array
    
    Args:
        video_frames: List/array of video frames (numpy arrays)
        fps: Frames per second for the output video
        filename: Optional filename (will generate UUID if not provided)
    
    Returns:
        str: Cloudinary secure URL
    """
    import uuid
    
    if filename is None:
        filename = f"deepfake_output_{str(uuid.uuid4())[:8]}.mp4"
    
    # Create video in memory using BytesIO
    buffer = io.BytesIO()
    
    # We need to create a temporary file for OpenCV VideoWriter
    # as it doesn't support writing directly to BytesIO
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Write video to temporary file
        height, width = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
        
        for frame in video_frames:
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        
        # Read the temporary file into memory
        with open(tmp_path, 'rb') as f:
            video_data = f.read()
        
        # Create BytesIO object
        video_buffer = io.BytesIO(video_data)
        video_buffer.name = filename  # Cloudinary needs a filename
        
        # Upload directly from BytesIO
        result = cloudinary.uploader.upload_large(
            video_buffer,
            resource_type='video',
            folder='deepfake-outputs',
            public_id=os.path.splitext(filename)[0]  # Remove extension
        )
        
        print(f"Uploaded to: {result['secure_url']}")
        return result['secure_url']
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def upload_from_bytes(video_bytes, filename=None):
    """
    Upload video directly to Cloudinary from bytes
    
    Args:
        video_bytes: Raw video file bytes
        filename: Optional filename for Cloudinary
    
    Returns:
        str: Cloudinary secure URL
    """
    import uuid
    
    if filename is None:
        filename = f"deepfake_output_{str(uuid.uuid4())[:8]}.mp4"
    
    # Create BytesIO object from bytes
    video_buffer = io.BytesIO(video_bytes)
    video_buffer.name = filename  # Cloudinary needs a filename
    
    # Upload directly from BytesIO
    result = cloudinary.uploader.upload_large(
        video_buffer,
        resource_type='video',
        folder='deepfake-outputs',
        public_id=os.path.splitext(filename)[0]  # Remove extension
    )
    
    print(f"Uploaded to: {result['secure_url']}")
    return result['secure_url']

# Function to download video from URL into memory
import requests

def download_video_to_memory(url, timeout=30, max_size_mb=100):
    """
    Download video from URL directly into memory
    
    Args:
        url: Video URL to download
        timeout: Request timeout in seconds
        max_size_mb: Maximum file size in MB to prevent memory issues
    
    Returns:
        bytes: Video content as bytes
    """
    try:
        # Stream the download to handle large files efficiently
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Check content length if available
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValueError(f"Video file too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")
        
        # Download in chunks to manage memory
        video_data = b""
        downloaded_size = 0
        max_bytes = max_size_mb * 1024 * 1024
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                downloaded_size += len(chunk)
                
                # Check size limit during download
                if downloaded_size > max_bytes:
                    raise ValueError(f"Video file too large during download (max: {max_size_mb}MB)")
                
                video_data += chunk
        
        if not video_data:
            raise ValueError("Downloaded video is empty")
            
        return video_data
        
    except requests.RequestException as e:
        raise Exception(f"Failed to download video from URL: {str(e)}")
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")


# Modified version of your original upload function for backwards compatibility
def upload(path):
    """Original upload function - kept for backwards compatibility"""
    result = cloudinary.uploader.upload_large(
        path,
        resource_type='video',
        folder='deepfake-outputs'
    )
    print(result['secure_url'])
    return result['secure_url']
