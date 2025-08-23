import io
import tempfile
import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras.models import Model

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
IMG_SIZE = 224

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video_from_bytes(video_bytes, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    # Create a temporary file in memory or use a very short-lived temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames)
    finally:
        # Clean up the temporary file immediately
        os.unlink(tmp_path)

# Alternative approach using BytesIO (may not work with OpenCV directly)
def load_video_from_bytesio(video_bytes, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    """
    Alternative approach - Note: OpenCV doesn't directly support BytesIO,
    so this is more complex and might require additional libraries like imageio
    """
    try:
        import imageio
        
        # Create a BytesIO object
        video_buffer = io.BytesIO(video_bytes)
        
        # Use imageio to read video from bytes
        reader = imageio.get_reader(video_buffer, 'mp4')
        frames = []
        
        for i, frame in enumerate(reader):
            if max_frames and len(frames) >= max_frames:
                break
                
            # Convert from RGB to BGR for consistency with OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
        
        reader.close()
        return np.array(frames)
    except ImportError:
        raise ImportError("imageio library is required for this approach. Install with: pip install imageio[ffmpeg]")

def build_feature_extractor():
    feature_extractor = applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    
    preprocess_input = applications.inception_v3.preprocess_input

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return Model(inputs, outputs, name='feature_extractor')

def prepare_single_video(frames):
    feature_extractor = build_feature_extractor()
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype='bool')
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1

    return frame_features, frame_mask

def sequence_prediction_from_bytes(video_bytes, model_path):
    """Main function that processes video from bytes"""
    model = tf.keras.models.load_model(model_path)
    
    # Method 1: Using temporary file (recommended for OpenCV)
    frames = load_video_from_bytes(video_bytes)
    
    # Method 2: Using imageio (alternative, requires additional dependency)
    # frames = load_video_from_bytesio(video_bytes)
    
    frame_features, frame_mask = prepare_single_video(frames)
    return model.predict([frame_features, frame_mask])[0]

# Modified DeepfakeVideoExplainability class to work with bytes
class DeepfakeVideoExplainabilityFromBytes:
    def __init__(self, sequence_model, feature_extractor=None):
        self.sequence_model = sequence_model
        self.feature_extractor = feature_extractor or build_feature_extractor()
        self.class_names = ["REAL", 'FAKE']

    def feature_importance_analysis(self, video_bytes):
        frames = load_video_from_bytes(video_bytes)
        original_features, original_mask = prepare_single_video(frames)
        
        baseline_pred = self.sequence_model.predict([original_features, original_mask])[0][0]
        
        frame_importance = []
        valid_frames = min(len(frames), MAX_SEQ_LENGTH)
        
        for i in range(valid_frames):
            modified_features = original_features.copy()
            modified_features[0, i, :] = 0  
            
            modified_mask = original_mask.copy()
            modified_mask[0, i] = False  
            
            modified_pred = self.sequence_model.predict([modified_features, modified_mask])[0][0]
            
            importance = abs(baseline_pred - modified_pred)
            frame_importance.append(importance)
            
        while len(frame_importance) < MAX_SEQ_LENGTH:
            frame_importance.append(0.0)
        
        return {
            'frame_importance': np.array(frame_importance),
            'baseline_prediction': baseline_pred,
            'predicted_class': 'FAKE' if baseline_pred >= 0.5 else 'REAL',
            'confidence': baseline_pred if baseline_pred >= 0.5 else 1 - baseline_pred,
            'valid_frames': valid_frames
        }

    def temporal_attention_analysis(self, video_bytes):
        frames = load_video_from_bytes(video_bytes)
        
        importance_result = self.feature_importance_analysis(video_bytes)
        frame_importance = importance_result["frame_importance"]
        
        valid_importance = frame_importance[:importance_result["valid_frames"]]
        if np.sum(valid_importance) > 0:
            attention_weights = tf.nn.softmax(valid_importance).numpy()
        else:
            attention_weights = np.ones(len(valid_importance)) / len(valid_importance)
        
        full_attention = np.zeros(MAX_SEQ_LENGTH)
        full_attention[:len(attention_weights)] = attention_weights
        
        return {
            'attention_weights': full_attention,
            'peak_frames': np.argsort(attention_weights)[-3:][::-1],
            'attention_entropy': -np.sum(attention_weights * np.log(attention_weights + 1e-10))
        }

    def feature_space_analysis(self, video_bytes):
        frames = load_video_from_bytes(video_bytes)
        frame_features, frame_mask = prepare_single_video(frames)
        
        baseline_pred = self.sequence_model.predict([frame_features, frame_mask])[0][0]
        
        feature_importance = np.zeros(NUM_FEATURES)
        valid_frames = min(len(frames), MAX_SEQ_LENGTH)
        
        sample_features = np.random.choice(NUM_FEATURES, size=min(100, NUM_FEATURES), replace=False)
        
        for feat_idx in sample_features:
            modified_features = frame_features.copy()
            modified_features[0, :valid_frames, feat_idx] = 0
            
            modified_pred = self.sequence_model.predict([modified_features, frame_mask])[0][0]
            
            importance = abs(baseline_pred - modified_pred)
            feature_importance[feat_idx] = importance
        
        return {
            'feature_importance': feature_importance,
            'top_features': np.argsort(feature_importance)[-10:][::-1],
            'feature_stats': {
                'mean': np.mean(feature_importance),
                'std': np.std(feature_importance),
                'max': np.max(feature_importance)
            }
        }

    def spatial_attention_analysis(self, video_bytes):
        frames = load_video_from_bytes(video_bytes)
        
        if not hasattr(self.feature_extractor, '_built') or not self.feature_extractor._built:
            dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            _ = self.feature_extractor(dummy_input)
        
        inception_base = None
        for layer in self.feature_extractor.layers:
            if hasattr(layer, 'name') and 'inception_v3' in layer.name.lower():
                inception_base = layer
                break
        
        if inception_base is None:
            for layer in self.feature_extractor.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 100:
                    inception_base = layer
                    break
        
        if inception_base is None:
            for layer in reversed(self.feature_extractor.layers):
                if hasattr(layer, 'layers'):
                    inception_base = layer
                    break
        
        if inception_base is None:
            return self._simplified_spatial_analysis(video_bytes)
        
        last_conv_layer = None
        conv_layer_names = ["mixed10", 'mixed9', 'mixed8', 'mixed7']
        
        for layer_name in conv_layer_names:
            try:
                last_conv_layer = inception_base.get_layer(layer_name)
                break
            except ValueError:
                continue
        
        if last_conv_layer is None:
            for layer in reversed(inception_base.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
        
        if last_conv_layer is None:
            return self._simplified_spatial_analysis(video_bytes)
        
        try:
            grad_model = tf.keras.models.Model(
                inputs=self.feature_extractor.input,
                outputs=[last_conv_layer.output, self.feature_extractor.output]
            )
            
            spatial_maps = []
            valid_frames = min(len(frames), MAX_SEQ_LENGTH)
            
            for i in range(valid_frames):
                frame = frames[i:i+1].astype(np.float32)
                
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(frame)
                    target_output = tf.reduce_mean(predictions)
                
                grads = tape.gradient(target_output, conv_outputs)
                
                if grads is None:
                    heatmap = np.random.rand(8, 8)
                else:
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
                    heatmap = tf.maximum(heatmap, 0)
                    
                    if tf.reduce_max(heatmap) > 0:
                        heatmap = heatmap / tf.reduce_max(heatmap)
                    
                    heatmap = heatmap.numpy()
                
                spatial_maps.append(heatmap)
            
            return {
                'spatial_maps': spatial_maps,
                'num_frames': valid_frames
            }
            
        except Exception as e:
            return self._simplified_spatial_analysis(video_bytes)

    def _simplified_spatial_analysis(self, video_bytes):
        frames = load_video_from_bytes(video_bytes)
        valid_frames = min(len(frames), MAX_SEQ_LENGTH)
        
        spatial_maps = []
        
        for i in range(valid_frames):
            frame = frames[i]
            
            gray_frame = np.mean(frame, axis=2)
            
            kernel_size = 16
            h, w = gray_frame.shape
            
            heatmap_h, heatmap_w = h // kernel_size, w // kernel_size
            heatmap = np.zeros((heatmap_h, heatmap_w))
            
            for y in range(heatmap_h):
                for x in range(heatmap_w):
                    y_start, y_end = y * kernel_size, (y + 1) * kernel_size
                    x_start, x_end = x * kernel_size, (x + 1) * kernel_size
                    
                    if y_end <= h and x_end <= w:
                        patch = gray_frame[y_start:y_end, x_start:x_end]
                        heatmap[y, x] = np.var(patch)
            
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            spatial_maps.append(heatmap)
        
        return {
            'spatial_maps': spatial_maps,
            'num_frames': valid_frames
        }

    def comprehensive_analysis(self, video_bytes, output_dir=None):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        feature_analysis = self.feature_importance_analysis(video_bytes)
        temporal_analysis = self.temporal_attention_analysis(video_bytes)
        feature_space = self.feature_space_analysis(video_bytes)
        spatial_analysis = self.spatial_attention_analysis(video_bytes)
        
        return {
            'feature_analysis': feature_analysis,
            'temporal_analysis': temporal_analysis,
            'feature_space': feature_space,
            'spatial_analysis': spatial_analysis
        }

    def create_overlay_video(self, video_bytes, spatial_analysis, output_path=None, fps=None, 
                            beyond_analysis_mode='last_map', return_bytes=False):
        
        if not spatial_analysis["spatial_maps"]:
            return None
        
        # Create temporary file just for getting video properties
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return None
            
            original_fps = cap.get(cv2.CAP_PROP_FPS) if fps is None else fps
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Load all frames from bytes
            full_frames = load_video_from_bytes(video_bytes, max_frames=0)
            
            spatial_maps = spatial_analysis["spatial_maps"]
            num_analysis_frames = len(spatial_maps)
            
            overlayed_frames = []
            
            for i, frame in enumerate(full_frames):
                frame_uint8 = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                
                spatial_map = None
                overlay_strength = 0.3
                
                if i < num_analysis_frames:
                    spatial_map = spatial_maps[i]
                    overlay_strength = 0.3
                else:
                    if beyond_analysis_mode == 'last_map':
                        spatial_map = spatial_maps[-1] if spatial_maps else None
                        overlay_strength = 0.15
                        
                    elif beyond_analysis_mode == 'fade_out':
                        fade_frames = 30
                        frames_beyond = i - num_analysis_frames
                        if frames_beyond < fade_frames:
                            fade_factor = max(0, 1 - frames_beyond / fade_frames)
                            spatial_map = spatial_maps[-1] * fade_factor if spatial_maps else None
                            overlay_strength = 0.3 * fade_factor
                        else:
                            spatial_map = None
                            
                    elif beyond_analysis_mode == 'repeat_analysis':
                        pattern_idx = i % num_analysis_frames
                        spatial_map = spatial_maps[pattern_idx]
                        overlay_strength = 0.2
                        
                    elif beyond_analysis_mode == 'no_overlay':
                        spatial_map = None
                
                if spatial_map is not None:
                    spatial_map_resized = cv2.resize(spatial_map, (IMG_SIZE, IMG_SIZE))
                    
                    heatmap = cv2.applyColorMap(
                        (spatial_map_resized * 255).astype(np.uint8),
                        cv2.COLORMAP_JET
                    )
                    
                    overlay = cv2.addWeighted(frame_uint8, 0.7, heatmap, overlay_strength, 0)
                    overlayed_frames.append(overlay)
                else:
                    overlayed_frames.append(frame_uint8)
            
            # If return_bytes is True, create video in memory and return bytes
            if return_bytes:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_output:
                    tmp_output_path = tmp_output.name
                
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(tmp_output_path, fourcc, original_fps, (IMG_SIZE, IMG_SIZE))
                    
                    for frame in overlayed_frames:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    out.release()
                    
                    # Read the video as bytes
                    with open(tmp_output_path, 'rb') as f:
                        video_bytes_output = f.read()
                    
                    return video_bytes_output
                    
                finally:
                    if os.path.exists(tmp_output_path):
                        os.unlink(tmp_output_path)
            
            # Original behavior - save to file if output_path is provided
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, original_fps, (IMG_SIZE, IMG_SIZE))
                
                for frame in overlayed_frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                out.release()
            
            return overlayed_frames
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

def explain_deepfake_video_from_bytes(sequence_model_path, video_bytes, output_dir=None, 
                                     create_overlay=True, overlay_mode='fade_out', return_overlay_bytes=False):
   
    sequence_model = tf.keras.models.load_model(sequence_model_path)
    explainer = DeepfakeVideoExplainabilityFromBytes(sequence_model)
    results = explainer.comprehensive_analysis(video_bytes, output_dir)
    
    # Get prediction results
    prediction = results["feature_analysis"]["predicted_class"]
    confidence = results["feature_analysis"]["confidence"]
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.3f}")
    
    if create_overlay:
        if return_overlay_bytes:
            # Return overlay video as bytes
            overlay_bytes = explainer.create_overlay_video(
                video_bytes, 
                results["spatial_analysis"], 
                output_path=None,
                beyond_analysis_mode=overlay_mode,
                return_bytes=True
            )
            
            results["overlay_video_bytes"] = overlay_bytes
            results["overlay_frames_created"] = len(explainer.create_overlay_video(
                video_bytes, results["spatial_analysis"], return_bytes=False
            )) if overlay_bytes else 0
            
        elif output_dir:
            # Original behavior - save to file
            import uuid
            video_name = f"video_{str(uuid.uuid4())[:8]}"
            overlay_output_path = os.path.join(output_dir, f'{video_name}_overlay.mp4')
            
            overlay_frames = explainer.create_overlay_video(
                video_bytes, 
                results["spatial_analysis"], 
                overlay_output_path,
                beyond_analysis_mode=overlay_mode
            )
            
            results["overlay_video_path"] = overlay_output_path
            results["overlay_frames_created"] = len(overlay_frames) if overlay_frames else 0
            
            print(f"Overlay video saved: {overlay_output_path}")
    
    return results