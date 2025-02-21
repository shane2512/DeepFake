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
        kwargs.pop('groups', None)  # Remove 'groups' to avoid errors
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)  # Ensure config doesn't contain 'groups'
        return super().from_config(config)

class CustomSeparableConv2D(tf.keras.layers.SeparableConv2D):
    def __init__(self, *args, **kwargs):
        for key in ['groups', 'kernel_initializer', 'kernel_regularizer', 'kernel_constraint']:
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        for key in ['groups', 'kernel_initializer', 'kernel_regularizer', 'kernel_constraint']:
            config.pop(key, None)
        return super().from_config(config)

# ---------------------------
# Model Paths (update as needed)
# ---------------------------
DEEPFAKE_MODEL_PATH =r"C:\Users\shane\Downloads\New folder\detection_model_video.h5"
AUTOENCODER_MODEL_PATH = r"C:\Users\shane\Downloads\New folder\detection_model_video.h5"

# ---------------------------
# Load Models
# ---------------------------
def safe_load_model(model_path):
    try:
        return load_model(
            model_path,
            custom_objects={
                'DepthwiseConv2D': CustomDepthwiseConv2D,
                'SeparableConv2D': CustomSeparableConv2D
            }
        )
    except TypeError as e:
        print(f"⚠️ TypeError while loading {model_path}: {e}")
    except ValueError as e:
        print(f"⚠️ ValueError while loading {model_path}: {e}")
    except Exception as e:
        print(f"⚠️ Unknown error while loading {model_path}: {e}")

    return None  # Return None if loading fails

# Load models safely
deepfake_model = safe_load_model(DEEPFAKE_MODEL_PATH)
autoencoder_model = safe_load_model(AUTOENCODER_MODEL_PATH)

# Check if models loaded successfully
if deepfake_model is None:
    print("❌ Error: Failed to load Deepfake model")
if autoencoder_model is None:
    print("❌ Error: Failed to load Autoencoder model")


# ---------------------------
# Flask App Initialization
# ---------------------------
app = Flask(__name__)
app.secret_key = 'nerain$1'  # Needed for session storage

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------
# Preprocessing: for images
# ---------------------------
def preprocess_image(frame_bgr):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_array = image_resized / 255.0
    return np.expand_dims(image_array, axis=0)

# ---------------------------
# Video Processing: Analyze up to 20 frames and use first frame for autoencoder
# ---------------------------
def process_video_first_20_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    predictions = []
    max_frames = 20
    frame_count = 0

    rep_input_file = f"{base_name}_representative.jpg"
    rep_recon_file = f"{base_name}_representative_recon.jpg"
    rep_input_path = os.path.join(UPLOAD_FOLDER, rep_input_file)
    rep_recon_path = os.path.join(UPLOAD_FOLDER, rep_recon_file)

    rep_frame_bgr = None  # store the first frame

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Save first frame as representative
        if frame_count == 1:
            rep_frame_bgr = frame
            cv2.imwrite(rep_input_path, rep_frame_bgr)

        # Predict using the deepfake detection model
        input_data = preprocess_image(frame)
        pred = deepfake_model.predict(input_data)[0][0]
        predictions.append(pred)

    cap.release()

    if len(predictions) == 0 or rep_frame_bgr is None:
        return None, None, None, None

    # Average prediction from the frames
    avg_pred = sum(predictions) / len(predictions)
    # If model outputs probability of being fake, then:
    if avg_pred > 0.5:
        detection_result = f"Fake detected (Confidence: {avg_pred*100:.2f}%)"
    else:
        detection_result = f"Real detected (Confidence: {(1-avg_pred)*100:.2f}%)"

    # Reconstruct representative frame using autoencoder
    rep_input_data = preprocess_image(rep_frame_bgr)
    rep_reconstructed = autoencoder_model.predict(rep_input_data)[0]
    rep_reconstructed = (rep_reconstructed * 255).astype(np.uint8)
    rep_reconstructed_bgr = cv2.cvtColor(rep_reconstructed, cv2.COLOR_RGB2BGR)
    cv2.imwrite(rep_recon_path, rep_reconstructed_bgr)

    return detection_result, rep_input_file, rep_recon_file, avg_pred

# ---------------------------
# Process File (Image or Video)
# ---------------------------
def process_file(file_path, file_type):
    base, _ = os.path.splitext(os.path.basename(file_path))

    if file_type == "image":
        image = cv2.imread(file_path)
        if image is None:
            return None, None, None

        input_data = preprocess_image(image)
        pred = deepfake_model.predict(input_data)[0][0]
        if pred > 0.5:
            detection_result = f"Fake detected (Confidence: {pred*100:.2f}%)"
        else:
            detection_result = f"Real detected (Confidence: {(1-pred)*100:.2f}%)"

        # Reconstruct image for visualization
        recon = autoencoder_model.predict(input_data)[0]
        recon = (recon * 255).astype(np.uint8)
        recon_bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)

        rep_input_file = f"{base}_representative.jpg"
        rep_recon_file = f"{base}_representative_recon.jpg"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, rep_input_file), image)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, rep_recon_file), recon_bgr)

        return detection_result, rep_input_file, rep_recon_file
    else:
        detection_result, rep_input_file, rep_recon_file, _ = process_video_first_20_frames(file_path)
        if detection_result is None:
            return None, None, None
        return detection_result, rep_input_file, rep_recon_file

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        file_type = "image" if file_ext in {'png', 'jpg', 'jpeg', 'gif'} else "video"
        
        detection_result, rep_input_file, rep_recon_file = process_file(file_path, file_type)
        if detection_result is None:
            return "Error processing file", 400
        
        return render_template(
            'result.html',
            prediction=detection_result,
            uploaded_file=rep_input_file,
            reconstructed_file=rep_recon_file,
            file_type=file_type
        )
    
    return "Invalid file type", 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)

