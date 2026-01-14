print("🚀 train.py started", flush=True)
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# This imports the create_model function from your model.py file
from model import create_model
import tensorflow as tf

print("TensorFlow version:", tf.__version__, flush=True)

# ===============================
# CONFIGURATION
# ===============================

# The model expects sequences of exactly 50 keystroke intervals
SEQUENCE_LENGTH = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")

# Define the paths to the four emotion datasets
FILES = {
    "happy": os.path.join(DATASET_DIR, "happy.csv"),
    "sad": os.path.join(DATASET_DIR, "sad.csv"),
    "calm": os.path.join(DATASET_DIR, "calm.csv"),
    "stressed": os.path.join(DATASET_DIR, "stressed.csv")
}

# ===============================
# LOAD DATA
# ===============================

X = []
y = []

print("📂 Loading dataset...")

for label, file_path in FILES.items():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    # Read the CSV file containing 50 columns of timing data
    df = pd.read_csv(file_path, header=None)

    for row in df.values:
        row = np.array(row, dtype=np.float32)

        # Remove NaNs if any exist in the data
        row = row[~np.isnan(row)]

        # Ensure every row is exactly 50 intervals long
        if len(row) >= SEQUENCE_LENGTH:
            row = row[:SEQUENCE_LENGTH]
        else:
            # Pad with zeros if the user typed fewer than 50 keys
            row = np.pad(row, (0, SEQUENCE_LENGTH - len(row)), mode="constant")

        X.append(row)
        y.append(label)

print(f"✅ Loaded {len(X)} samples")

# ===============================
# PREPARE DATA
# ===============================

# Reshape data to (samples, time_steps, features) for the LSTM layer
X = np.array(X).reshape(-1, SEQUENCE_LENGTH, 1)

# Convert text labels (happy, sad, etc.) into numerical categories
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)

print("📊 Data shape:", X.shape, y.shape)

# ===============================
# CREATE & TRAIN MODEL
# ===============================

# Initialize the model architecture defined in model.py
model = create_model()

# Stop training early if the loss stops improving to prevent overfitting
early_stop = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

print("🧠 Training LSTM model...")

# Train the model on the keystroke data
model.fit(
    X,
    y,
    epochs=20,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# SAVE MODEL
# ===============================

# Save the trained model to a file that the Flask app can load
MODEL_PATH = os.path.join(BASE_DIR, "emotion_lstm.h5")
model.save(MODEL_PATH)

print("🎉 Training complete!")
print(f"💾 Model saved as: {MODEL_PATH}")