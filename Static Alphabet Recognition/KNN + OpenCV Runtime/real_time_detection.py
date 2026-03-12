import cv2
import numpy as np
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
import joblib  # for loading the saved KNN model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load your trained KNN model
knn = joblib.load('knn_model.pkl')

def extract_hand_landmarks(image, hands):
    # Process image for landmarks
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    if results.multi_hand_landmarks:
        # Take the first detected hand only
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    else:
        return None

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            landmarks = extract_hand_landmarks(frame, hands)

            if landmarks is not None:
                # KNN expects 2D array
                pred = knn.predict([landmarks])[0]

                # Draw landmarks on frame
                mp_drawing.draw_landmarks(frame, hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                # Show prediction on screen
                cv2.putText(frame, f'Prediction: {pred}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow('Real-Time Sign Language Detection - Press Q to quit', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
