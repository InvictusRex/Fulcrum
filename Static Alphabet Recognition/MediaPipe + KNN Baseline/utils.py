import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

tts_engine = pyttsx3.init()

def extract_hand_landmarks(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords)
        else:
            return None

def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def draw_landmarks(image, landmarks):
    if landmarks is not None:
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
