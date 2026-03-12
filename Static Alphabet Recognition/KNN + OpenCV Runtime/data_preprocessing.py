import os
import cv2
import numpy as np
from utils import extract_hand_landmarks

DATASET_PATH = "../dataset/Gesture Image Data"  # Use raw images
features = []
labels = []

print("🟡 Starting preprocessing...")

for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        continue

    print(f"🔤 Processing label: {label}")
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lm = extract_hand_landmarks(img_rgb)
        if lm is not None:
            features.append(lm)
            labels.append(label)

features = np.array(features)
labels = np.array(labels)
np.save("features.npy", features)
np.save("labels.npy", labels)
print(f"✅ Saved {len(features)} samples.")
