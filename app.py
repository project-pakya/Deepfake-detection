app.py

# app.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
from efficientnet_pytorch import EfficientNet
from werkzeug.utils import secure_filename

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the model class
class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetectionModel, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# Load the model
def load_model():
    model = DeepfakeDetectionModel().to(device)
    
    # If you have a pre-trained model file, uncomment these lines and provide the path
    # model_path = 'path_to_your_model.pth'
    # model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    return model

# Initialize model
model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract frames from video
def extract_frames(video_path, num_frames=20):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        return None
    
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
    
    # Calculate interval to extract evenly distributed frames
    interval = total_frames // min(num_frames, total_frames)
    interval = max(1, interval)
    
    frame_indices = list(range(0, total_frames, interval))[:num_frames]
    
    for frame_idx in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = vidcap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    vidcap.release()
    return frames

# Function to predict if a frame is real or fake
def predict_frame(frame, model):
    image = Image.fromarray(frame)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        fake_probability = probabilities[0][1].item()  # Assuming class 1 is fake
    
    return fake_probability

# Function to predict if a video is real or fake
def predict_video(video_path, model):
    frames = extract_frames(video_path)
    
    if frames is None or len(frames) == 0:
        return {"error": "Could not extract frames from video"}
    
    # Get predictions for each frame
    probabilities = [predict_frame(frame, model) for frame in frames]
    
    # Average probability across frames
    avg_probability = sum(probabilities) / len(probabilities)
    
    # Classify video as fake if average probability > 0.5
    is_fake = avg_probability > 0.5
    
    return {
        "is_fake": bool(is_fake),
        "confidence": float(avg_probability if is_fake else 1 - avg_probability),
        "frame_count": len(frames)
    }

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling video upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'})
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No video selected'})
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    
    # Make prediction
    try:
        result = predict_video(video_path, model)
        result['video_path'] = video_path
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)