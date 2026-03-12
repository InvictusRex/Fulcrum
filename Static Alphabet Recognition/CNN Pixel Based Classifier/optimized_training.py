import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_dataset


IMG_SIZE = 64  # must be consistent with utils

def build_cnn_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model(model, test_ds, class_names):
    # Get true labels and predictions
    y_true = []
    y_pred = []
    y_pred_prob = []

    for images, labels in test_ds:
        preds = model.predict(images)
        preds_labels = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds_labels)
        y_pred_prob.extend(preds)  # store probabilities for ROC-AUC

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = np.mean(y_true == y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")

    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    try:
        roc_auc = roc_auc_score(y_true_onehot, y_pred_prob, average='weighted', multi_class='ovr')
        print(f"ROC-AUC (weighted, OVR): {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC calculation error: {e}")

def main():
    # Print TensorFlow device info
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPU devices:", tf.config.list_physical_devices('GPU'))

    DATA_DIR = r'E:\1_Work_Files\13_Project - S1gnvers3\S1gnvers3\dataset\Dataset_Split\train'
    VAL_DIR = r'E:\1_Work_Files\13_Project - S1gnvers3\S1gnvers3\dataset\Dataset_Split\val'
    TEST_DIR = r'E:\1_Work_Files\13_Project - S1gnvers3\S1gnvers3\dataset\Dataset_Split\test'

    print("Loading datasets...")
    train_ds = load_dataset(DATA_DIR)
    val_ds = load_dataset(VAL_DIR)
    test_ds = load_dataset(TEST_DIR)

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    model = build_cnn_model(len(class_names))

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20
    )

    print("Training complete. Saving model...")
    model.save('cnn_asl_model')

    print("\nEvaluating on test dataset...")
    evaluate_model(model, test_ds, class_names)


if __name__ == "__main__":
    main()