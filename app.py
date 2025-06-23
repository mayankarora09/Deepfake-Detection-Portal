import os
import torch
import librosa
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model as load_audio_model
from torchvision import transforms
from PIL import Image

# === INIT FLASK APP === #
app = Flask(__name__, template_folder='templates', static_folder='static')

# === DEVICE SETUP === #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === LOAD MODELS === #

# Audio model
audio_model = load_audio_model('final_deepfake_audio_model.h5')

# Image model
from torchvision.models import resnet18
import torch.nn as nn
image_model = resnet18(pretrained=False)
image_model.fc = nn.Linear(image_model.fc.in_features, 2)
image_model.load_state_dict(torch.load('best_deepfake_detector.pth', map_location=device))
image_model.to(device)
image_model.eval()

# Video model
from video_model import CNNRNNModel  # Your custom class
video_model = CNNRNNModel()
video_model.load_state_dict(torch.load('best_resnet_lstm_model.pth', map_location=device))
video_model.to(device)
video_model.eval()

# === UTILITY FUNCTIONS === #

def predict_audio(file_path):
    try:
        print(f"Processing WAV file: {file_path}")
        y, sr = librosa.load(file_path, sr=22050, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = np.pad(mfcc, ((0, 0), (0, max(0, 174 - mfcc.shape[1]))), mode='constant')
        mfcc = mfcc[:, :174]
        mfcc = mfcc[..., np.newaxis][np.newaxis, ...]  # shape: (1, 40, 174, 1)

        pred = audio_model.predict(mfcc)[0][0]
        print(f"Audio prediction score: {pred}")
        return "FAKE" if pred > 0.5 else "REAL"
    except Exception as e:
        print(f"[ERROR] Audio prediction failed: {e}")
        return "Error processing WAV file"

def predict_image(file_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = image_model(image)
        pred = torch.argmax(output, dim=1).item()
    return "FAKE" if pred == 0 else "REAL"

def predict_video(file_path, max_frames=50):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    cap = cv2.VideoCapture(file_path)
    frames = []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
        image = transform(image)
        frames.append(image)
        count += 1

    cap.release()

    if len(frames) == 0:
        return "Error: Could not read video frames."

    if len(frames) < max_frames:
        frames += [frames[-1]] * (max_frames - len(frames))

    video_tensor = torch.stack(frames).unsqueeze(0).to(device)  # Shape: [1, T, C, H, W]
    with torch.no_grad():
        output = video_model(video_tensor)
        pred = torch.argmax(output, dim=1).item()

    return "FAKE" if pred == 1 else "REAL"

# === ROUTES === #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    os.makedirs('uploads', exist_ok=True)
    save_path = os.path.join('uploads', filename)
    file.save(save_path)

    print(f"File received: {filename} (.{ext})")

    try:
        if ext in ['.wav']:
            result = predict_audio(save_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            result = predict_image(save_path)
        elif ext in ['.mp4', '.avi', '.mov']:
            result = predict_video(save_path)
        else:
            return jsonify({'error': f'Unsupported file type: {ext}'}), 400

        return jsonify({'result': result})

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({'error': 'Internal error occurred'}), 500

# === START APP === #
if __name__ == '__main__':
    app.run(debug=True)
