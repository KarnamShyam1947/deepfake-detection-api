import time
import json
import mimetypes
from functools import wraps
from flask_cors import CORS
from flask import Flask, after_this_request
from utils.log_utils import configure_logging
from werkzeug.datastructures import FileStorage
from utils.video_utils import download_video, delete_file
from flask_restx import Api, Resource, Namespace, reqparse
from utils.prediction_utils import sequence_prediction, explain_deepfake_video
from utils.upload_utils import upload_from_bytes, download_video_to_memory, upload
from utils.v2.prediction_utils import sequence_prediction_from_bytes, explain_deepfake_video_from_bytes

configure_logging()

app = Flask(__name__)
CORS(app)

api = Api(
    app=app,
    version="1.0",
    title="Deepfake Detection API",
    description="Deepfake detection endpoints"
)

def measure_execution_time(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        start_time = time.perf_counter()

        @after_this_request
        def track_response(response):
            elapsed = time.perf_counter() - start_time
            if response.is_json:
                data = response.get_json()
                data["execution_time"] = f"{elapsed:.2f} sec."
                response.set_data(
                    app.response_class(
                        response=json.dumps(data),
                        status=response.status_code,
                        mimetype="application/json"
                    ).get_data()
                )
            return response

        return f(*args, **kwargs)
    return wrapped

ALLOWED_VIDEO_MIME_TYPES = {
    "video/mp4", "video/avi", "video/mpeg", "video/quicktime", "video/x-matroska"
}

def is_allowed_video(file: FileStorage):
    mime_type, _ = mimetypes.guess_type(file.filename)
    return mime_type in ALLOWED_VIDEO_MIME_TYPES

video_request = reqparse.RequestParser()
video_request.add_argument(
    name="url",
    location="json",
    type=str,
    required=True,
    help="Video URL required..."
)

video_upload_request = reqparse.RequestParser()
video_upload_request.add_argument(
    name="video",
    type=FileStorage,
    location="files",
    required=True,
    help="Only video files are allowed."
)

predict_ns = Namespace(
    name="Predict V1",
    description="The video will be stored locally for the prediction to happen",
    path="/api/v1/predict"
)

predict_v2_ns = Namespace(
    name="Predict V2",
    description="The video will not be stored locally for the prediction, using bytes only",
    path="/api/v2/predict"
)

@predict_ns.route("/")
class PredClass(Resource):
    @predict_ns.expect(video_request)
    @measure_execution_time
    def post(self):
        args = video_request.parse_args()
        video = args["url"]
        filepath = download_video(video)

        model_path = "models/video.keras"
        score = sequence_prediction(filepath, model_path)
        predicted_class = "FAKE" if score >= 0.5 else "REAL"
        confidence = score if score >= 0.5 else 1 - score
        confidence_percent = f"{confidence[0] * 100:.2f} %"

        result = explain_deepfake_video(
            sequence_model_path=model_path,
            video_path=filepath,
            output_dir="output",
            create_overlay=True,
            overlay_mode="repeat_analysis"
        )

        output_path = result["overlay_video_path"]
        video_url = upload(output_path)

        delete_file(filepath)
        delete_file(output_path)

        # Return as JSON; execution_time will be added by decorator
        return {
            "output_url": video_url,
            "predicted_class": predicted_class,
            "confidence": confidence_percent
        }

@predict_ns.route("/file")
class PrClass(Resource):
    @predict_ns.expect(video_upload_request)
    @measure_execution_time
    def post(self):
        args = video_upload_request.parse_args()
        video_src: FileStorage = args["video"]

        if not is_allowed_video(video_src):
            return {"message": "Invalid file type. Only video files are allowed."}, 400

        filepath = "temp_video.mp4"
        video_src.save(filepath)

        model_path = "models/video.keras"
        score = sequence_prediction(filepath, model_path)
        predicted_class = "FAKE" if score >= 0.5 else "REAL"
        confidence = score if score >= 0.5 else 1 - score
        confidence_percent = f"{confidence[0] * 100:.2f} %"

        result = explain_deepfake_video(
            sequence_model_path=model_path,
            video_path=filepath,
            output_dir="output",
            create_overlay=True,
            overlay_mode="repeat_analysis"
        )

        output_path = result["overlay_video_path"]
        video_url = upload(output_path)

        delete_file(filepath)
        delete_file(output_path)

        return {
            "output_url": video_url,
            "predicted_class": predicted_class,
            "confidence": confidence_percent
        }


@predict_v2_ns.route("/")
class PredV2Class(Resource):
    @predict_ns.expect(video_request)
    @measure_execution_time
    def post(self):
        args = video_request.parse_args()
        video_url = args["url"]
        
        try:
            # Download video directly into memory
            video_bytes = download_video_to_memory(video_url)
            
            # Process video from memory
            model_path = "models/video.keras"
            score = sequence_prediction_from_bytes(video_bytes, model_path)
            predicted_class = "FAKE" if score >= 0.5 else "REAL"
            confidence = score if score >= 0.5 else 1 - score
            confidence_percent = f"{confidence[0] * 100:.2f} %"

            # Use the bytes-based explanation function with direct memory return
            result = explain_deepfake_video_from_bytes(
                sequence_model_path=model_path,
                video_bytes=video_bytes,
                output_dir=None,  # No need for output directory
                create_overlay=True,
                overlay_mode="repeat_analysis",
                return_overlay_bytes=True  # Return overlay as bytes instead of saving to file
            )

            # Upload directly from memory to Cloudinary
            overlay_video_bytes = result["overlay_video_bytes"]
            video_url_result = upload_from_bytes(overlay_video_bytes)

            # No files to delete since everything was done in memory!

            return {
                "output_url": video_url_result,
                "predicted_class": predicted_class,
                "confidence": confidence_percent
            }
            
        except Exception as e:
            return {
                "error": f"Failed to process video: {str(e)}"
            }, 400

@predict_v2_ns.route("file")
class PrClass(Resource):
    @predict_ns.expect(video_upload_request)
    @measure_execution_time
    def post(self):
        args = video_upload_request.parse_args()
        video_src: FileStorage = args["video"]

        if not is_allowed_video(video_src):
            return {"message": "Invalid file type. Only video files are allowed."}, 400

        video_bytes = video_src.read()
        video_src.seek(0)  # Reset file pointer in case needed elsewhere

        model_path = "models/video.keras"
        score = sequence_prediction_from_bytes(video_bytes, model_path)
        predicted_class = "FAKE" if score >= 0.5 else "REAL"
        confidence = score if score >= 0.5 else 1 - score
        confidence_percent = f"{confidence[0] * 100:.2f} %"
        
        result = explain_deepfake_video_from_bytes(
            sequence_model_path=model_path,
            video_bytes=video_bytes,
            output_dir=None,  # No need for output directory
            create_overlay=True,
            overlay_mode="repeat_analysis",
            return_overlay_bytes=True  # Return overlay as bytes instead of saving to file
        )

        overlay_video_bytes = result["overlay_video_bytes"]
        video_url = upload_from_bytes(overlay_video_bytes)

        return {
            "output_url": video_url,
            "predicted_class": predicted_class,
            "confidence": confidence_percent
        }
 

api.add_namespace(predict_ns)
api.add_namespace(predict_v2_ns)

if __name__ == "__main__":
    app.run(port=5050, host="0.0.0.0")
