import cv2
import tensorflow as tf
import numpy as np

IMG_SIZE = 64
MODEL_PATH = 'cnn_asl_model'

class_names = None

def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    global class_names
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = model.class_names if hasattr(model, 'class_names') else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = preprocess_frame(img)

        preds = model.predict(input_img)
        pred_label = np.argmax(preds)
        confidence = np.max(preds)

        label_text = f"{class_names[pred_label]}: {confidence:.2f}" if class_names else f"Class: {pred_label} ({confidence:.2f})"

        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('ASL Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()