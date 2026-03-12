import streamlit as st
import cv2
import numpy as np
import pyttsx3
import os
import tempfile
from utils import extract_hand_landmarks
from sklearn.neighbors import KNeighborsClassifier

# Set env to suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load saved data
features = np.load('features.npy')
labels = np.load('labels.npy')

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features, labels)

# Setup Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Streamlit UI
st.set_page_config(page_title="S1gnvers3 - ASL Real-Time Recognition", layout="wide")
st.title("🤟 S1gnvers3: Real-Time ASL Gesture Detection")
st.markdown("Detects ASL alphabets, forms words, and speaks them out.")

col1, col2 = st.columns(2)
run = col1.checkbox('Start Camera')
speak = col1.button("🗣️ Speak Sentence")
clear = col2.button("🧹 Clear Buffer")

FRAME_WINDOW = st.image([])
sentence_display = st.empty()

buffer = []

# Open webcam
cap = None
if run:
    cap = cv2.VideoCapture(0)

while run and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not found or not working.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = extract_hand_landmarks(frame_rgb)

    if landmarks is not None:
        pred = knn.predict([landmarks])[0]

        # If gesture is 'fist', treat as space
        if pred.lower() == 'fist':
            if buffer and buffer[-1] != ' ':
                buffer.append(' ')
        else:
            if not buffer or buffer[-1] != pred:
                buffer.append(pred)

        sentence_display.markdown(f"### 📝 Detected: `{pred}`")

    # Draw frame
    FRAME_WINDOW.image(frame_rgb)

    # Update sentence
    sentence = ''.join(buffer).strip()
    st.markdown(f"## ✏️ Sentence: `{sentence}`")

    # Handle buttons
    if speak:
        engine.say(sentence)
        engine.runAndWait()
    if clear:
        buffer.clear()
        st.experimental_rerun()

# Cleanup
if cap:
    cap.release()
