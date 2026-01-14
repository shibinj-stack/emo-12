import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")

os.makedirs(DATASET_DIR, exist_ok=True)

emotions = {
    "happy": (20, 40),
    "sad": (60, 120),
    "calm": (40, 70),
    "stressed": (15, 90)
}

SAMPLES = 50      # rows per emotion
SEQUENCE_LEN = 50

for emotion, (low, high) in emotions.items():
    path = os.path.join(DATASET_DIR, f"{emotion}.csv")
    with open(path, "w") as f:
        for _ in range(SAMPLES):
            row = np.random.randint(low, high, SEQUENCE_LEN)
            f.write(",".join(map(str, row)) + "\n")

    print(f"✅ Created {path}")

print("🎉 Dataset generation complete!")