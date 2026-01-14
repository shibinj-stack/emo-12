import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Flask App Setup
app = Flask(__name__)
CORS(app)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_lstm.h5")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

model = None
emotions = ["Happy", "Sad", "Calm", "Stressed"]

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict():
    global model
    json_data = request.json
    keystroke_data = json_data.get("data", [])
    user_text = json_data.get("text", "")

    # --- STAGE 1: VADER NLP (Handles "not", "but", and combined emotions) ---
    if user_text.strip():
        vs = analyzer.polarity_scores(user_text)
        compound = vs['compound'] 

        # VADER correctly identifies "not happy" as negative
        if compound >= 0.4:
            return jsonify({"emotion": "Happy (NLP)", "confidence": float(compound)})
        elif compound <= -0.4:
            # If "stressed" is in the text and sentiment is negative, it returns Sad/Stressed context
            return jsonify({"emotion": "Sad/Stressed (NLP)", "confidence": float(abs(compound))})

    # --- STAGE 2: AI KEYSTROKE ANALYSIS FALLBACK ---
    if model is None:
        if not os.path.exists(MODEL_PATH):
            return jsonify({"emotion": "AI Model not found", "confidence": 0})
        model = load_model(MODEL_PATH)

    if not isinstance(keystroke_data, list) or len(keystroke_data) < 10:
        return jsonify({"emotion": "Neutral / Need more typing", "confidence": 0})

    # Prepare timing data
    data = np.array(keystroke_data, dtype=np.float32)
    data = np.pad(data, (0, max(0, 50 - len(data))))[:50].reshape(1, 50, 1)

    prediction = model.predict(data, verbose=0)
    confidence = float(np.max(prediction))
    emotion = emotions[int(np.argmax(prediction))]

    return jsonify({"emotion": emotion, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)