import os
import torch
import cv2
import numpy as np
import snntorch as snn  # Import SNN Torch
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for frontend communication
from moviepy.editor import VideoFileClip

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Setup Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Define a Simple Spiking Neural Network (SNN) Model
class SimpleSNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1024, 1)  # Fully connected layer (adjust input size)
        self.lif = snn.Leaky(beta=0.9)  # Spiking neuron

        # Initialize weights randomly
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = torch.tanh(self.fc(x))  # Normalize values
        return self.lif(x)  # Pass through SNN neuron

# ✅ Initialize the SNN Model
snn_model = SimpleSNN()

# ✅ Function to Extract Features from Video
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized = cv2.resize(gray, (32, 32))  # Resize to 32x32 for processing
        frames.append(resized.flatten())  # Flatten the frame

    cap.release()
    
    frames = np.array(frames)

    # Take the average pixel values as input to SNN
    if frames.size > 0:
        return torch.tensor(frames.mean(axis=0) / 255.0, dtype=torch.float32)
    else:
        return torch.zeros(1024)  # Return zeros if no frames are extracted

# ✅ API Endpoint to Upload and Analyze Video
@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Extract features from video
    input_data = extract_video_features(file_path)
    input_data = input_data.unsqueeze(0)  # Reshape for model

    # ✅ Pass through SNN Model
    output, _ = snn_model(input_data)

    return jsonify({"result": f"Processed by SNN: {output.item():.4f}"}), 200

# ✅ Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)
    
