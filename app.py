import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# ---------------------------
# Custom Layers (for model compatibility)
# ---------------------------
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)
        return super(CustomDepthwiseConv2D, cls).from_config(config)

class CustomSeparableConv2D(tf.keras.layers.SeparableConv2D):
    def __init__(self, *args, **kwargs):
        for key in ['groups', 'kernel_initializer', 'kernel_regularizer', 'kernel_constraint']:
            kwargs.pop(key, None)
        super(CustomSeparableConv2D, self).__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        for key in ['groups', 'kernel_initializer', 'kernel_regularizer', 'kernel_constraint']:
            config.pop(key, None)
        return super(CustomSeparableConv2D, cls).from_config(config)

# ---------------------------
# Model Paths
# ---------------------------
DEEPFAKE_MODEL_PATH = r"C:\Users\shane\Downloads\DF\detection_model_video.h5"
AUTOENCODER_MODEL_PATH = r"C:\Users\shane\Downloads\DF\detection_model_video.h5"

# ---------------------------
# Load Models
# ---------------------------
deepfake_model = load_model(
    DEEPFAKE_MODEL_PATH,
    custom_objects={
        'DepthwiseConv2D': CustomDepthwiseConv2D,
        'SeparableConv2D': CustomSeparableConv2D
    }
)
autoencoder_model = load_model(AUTOENCODER_MODEL_PATH)

# ---------------------------
# Flask App Initialization
# ---------------------------
app = Flask(__name__)
app.secret_key = 'nerain$1'  # Needed for session storage

# Folder configuration for uploads and lookup videos
UPLOAD_FOLDER = r'static/uploads'
VIDEO_FOLDER = r'/project/workspace/alt+hackj/df/Real'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_image(frame_bgr):
    """Resize and normalize BGR frame to (1, 256, 256, 3)."""
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_array = image_resized / 255.0
    return np.expand_dims(image_array, axis=0)

# ---------------------------
# Extract Middle Frame from Video
# ---------------------------
def extract_middle_frame(video_path):
    """Return the middle frame of the given video, or None if reading fails."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# ---------------------------
# Video Processing: Use only the Middle Frame
# ---------------------------
def process_video_middle_frame(video_path):
    """
    1. Extract the middle frame of the uploaded video (original_frame).
    2. Run deepfake detection on that frame.
    3. Return detection_result, filenames of original_frame.jpg and reconstructed_frame.jpg.
    """
    # Extract middle frame from uploaded video
    middle_frame = extract_middle_frame(video_path)
    if middle_frame is None:
        return None, None, None, None

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Save the original (middle) frame as an image
    original_file = f"{base_name}_middle.jpg"
    original_path = os.path.join(UPLOAD_FOLDER, original_file)
    cv2.imwrite(original_path, middle_frame)

    # Predict deepfake on this middle frame
    input_data = preprocess_image(middle_frame)
    pred = deepfake_model.predict(input_data)[0][0]

    # If fake detected
    if pred > 0.5:
        detection_result = f"Deepfake detected (Confidence: {pred * 100:.2f}%)"
        # Try to find a video with the same name in VIDEO_FOLDER
        matching_video_path = os.path.join(VIDEO_FOLDER, base_name + '.mp4')
        if os.path.exists(matching_video_path):
            # Extract middle frame from that "real" video
            replacement_frame = extract_middle_frame(matching_video_path)
            if replacement_frame is not None:
                reconstructed_file = f"{base_name}_recon.jpg"
                reconstructed_path = os.path.join(UPLOAD_FOLDER, reconstructed_file)
                cv2.imwrite(reconstructed_path, replacement_frame)
                return detection_result, original_file, reconstructed_file, pred
        
        # If no matching video found or extraction failed, fallback to autoencoder
        rep_reconstructed = autoencoder_model.predict(input_data)[0]
        rep_reconstructed = (rep_reconstructed * 255).astype(np.uint8)
        rep_reconstructed_bgr = cv2.cvtColor(rep_reconstructed, cv2.COLOR_RGB2BGR)
        reconstructed_file = f"{base_name}_recon.jpg"
        reconstructed_path = os.path.join(UPLOAD_FOLDER, reconstructed_file)
        cv2.imwrite(reconstructed_path, rep_reconstructed_bgr)
        return detection_result, original_file, reconstructed_file, pred

    else:
        # Real (not fake)
        detection_result = f"No deepfake detected (Confidence: {(1 - pred) * 100:.2f}%)"
        rep_reconstructed = autoencoder_model.predict(input_data)[0]
        rep_reconstructed = (rep_reconstructed * 255).astype(np.uint8)
        rep_reconstructed_bgr = cv2.cvtColor(rep_reconstructed, cv2.COLOR_RGB2BGR)
        reconstructed_file = f"{base_name}_recon.jpg"
        reconstructed_path = os.path.join(UPLOAD_FOLDER, reconstructed_file)
        cv2.imwrite(reconstructed_path, rep_reconstructed_bgr)
        return detection_result, original_file, reconstructed_file, pred

# ---------------------------
# Process File (Image or Video)
# ---------------------------
def process_file(file_path, file_type):
    """
    If it's an image, we process it directly.
    If it's a video, we only process its middle frame.
    Returns (detection_result, original_frame.jpg, reconstructed_frame.jpg).
    """
    base, _ = os.path.splitext(os.path.basename(file_path))
    
    if file_type == "image":
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            return None, None, None
        
        # Predict
        input_data = preprocess_image(image)
        pred = deepfake_model.predict(input_data)[0][0]
        if pred > 0.5:
            detection_result = f"Deepfake detected (Confidence: {pred*100:.2f}%)"
        else:
            detection_result = f"No deepfake detected (Confidence: {(1-pred)*100:.2f}%)"
        
        # Reconstruct via autoencoder
        recon = autoencoder_model.predict(input_data)[0]
        recon = (recon * 255).astype(np.uint8)
        recon_bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)

        # Save both frames
        original_file = f"{base}_original.jpg"
        reconstructed_file = f"{base}_recon.jpg"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, original_file), image)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, reconstructed_file), recon_bgr)

        return detection_result, original_file, reconstructed_file

    else:
        # It's a video: only process the middle frame
        detection_result, original_file, reconstructed_file, _ = process_video_middle_frame(file_path)
        return detection_result, original_file, reconstructed_file

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file uploads:
    1. Save the file in UPLOAD_FOLDER.
    2. Determine if it's an image or video.
    3. Process accordingly (detect deepfake, get frames).
    4. Render result.html with the detection info and frames.
    """
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Determine if it's an image or video
        file_ext = filename.rsplit('.', 1)[1].lower()
        file_type = "image" if file_ext in {'png', 'jpg', 'jpeg', 'gif'} else "video"
        
        # Process the file accordingly
        detection_result, original_frame, reconstructed_frame = process_file(file_path, file_type)
        if detection_result is None:
            return "Error processing file", 400
        
        # Render result.html, passing in the single-frame images
        return render_template(
            'result.html',
            prediction=detection_result,
            original_frame=original_frame,
            reconstructed_frame=reconstructed_frame
        )
    
    return "Invalid file type", 400

if __name__ == '__main__':
    app.run(debug=True)
