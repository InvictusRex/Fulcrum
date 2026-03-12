import os
import cv2
import numpy as np
import pandas as pd
from utils import extract_hand_landmarks

# Use a correct relative path
DATASET_PATH = "../dataset/Gesture Image Data"  # Adjust as needed from current working directory

def process_dataset_and_save_features():
    data = []
    labels = []

    print("Starting preprocessing...")

    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue

        print(f"Processing label: {label}")
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image {img_path}")
                continue

            landmarks = extract_hand_landmarks(img)
            if landmarks is not None:
                data.append(landmarks)
                labels.append(label)

    np.save("features.npy", np.array(data))
    np.save("labels.npy", np.array(labels))
    print(f"✅ Preprocessing complete: Saved {len(data)} samples to features.npy and labels.npy")

if __name__ == "__main__":
    process_dataset_and_save_features()
