import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_knn():
    X = np.load('features.npy')
    y = np.load('labels.npy')

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    joblib.dump(knn, 'model/knn_model.pkl')
    print("Model saved to model/knn_model.pkl")

if __name__ == "__main__":
    train_knn()