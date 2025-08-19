
# Deepfake Video Detection 

A Python service that processes input videos to perform predictions, generate Grad‑CAM‐based heatmap overlays, and uploads the resulting video to Cloudinary.

## Description

This API accepts a video, runs inference (e.g., classification or detection), overlays Grad‑CAM heatmaps on the output frames, reconstructs the video with these overlays, and uploads it to Cloudinary. The response includes URLs for accessing the processed video.

## Features

- Accepts video input via HTTP (e.g., multipart form).
- Performs model inference on frames.
- Visualizes model attention using Grad‑CAM.
- Reassembles frames into a heatmap-overlay video.
- Uploads the result to Cloudinary using `upload` or `upload_large` for files over 100 MB :contentReference.
- Returns a secure URL of the processed video.

## Workflow Summary

1. **Receive** video input.
2. **Extract frames** and run model predictions.
3. **Generate Grad‑CAM heatmap** for each frame to highlight model-attended areas (based on gradient-weighted localization technique).
4. **Overlay heatmap** onto each frame, then recompile the video.
5. **Upload result** to Cloudinary using the Python SDK—use `upload_large` for large video files.
6. **Return** the secure URL of the uploaded video.



## Sample Request

```http
POST http://localhost:5050/api/v1/predict HTTP/1.1
Content-Type: application/json

{
  "url": "https://example.com/path/to/input_video.mp4"
}
```

---

## Sample Response

```json
{
  "predicted_class": "REAL",
  "confidence": 92.34%,
  "output_url": "https://res.cloudinary.com/your-cloud/video/upload/v1234567890/heatmap_video.mp4",
  "execution_time": "12.34 sec."
}
```

---

### Explanation

* **Request**: You send a `POST` request to your API’s endpoint (for example `/analyze`) with a JSON payload containing the video’s accessible URL.
* **Response**:

  * `"predicted_class"` indicates the model’s label (e.g., `"REAL"` or `"FAKE"`).
  * `"confidence"` represents the model’s confidence score (typically a float between 0 and 100%).
  * `"output_url"` provides the secure Cloudinary URL where the processed video with the Grad‑CAM overlay is hosted.

This format ensures clarity and aligns well with standard RESTful API practices.


