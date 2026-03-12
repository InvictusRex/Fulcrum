import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
X = np.load('features.npy')
y = np.load('labels.npy')

# Optional: normalize features
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create MLP model
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)

# Train
print("Training MLP model...")
mlp.fit(X_train, y_train)

# Evaluate
y_pred = mlp.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(mlp, 'mlp_model.pkl')
print("Model saved as mlp_model.pkl")
