import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(img):
    results = hands.process(img)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    return None
