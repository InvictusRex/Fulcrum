import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

features = np.load('features.npy')
labels = np.load('labels.npy')

model = KNeighborsClassifier(n_neighbors=3)
model.fit(features, labels)

joblib.dump(model, 'knn_model.pkl')
print("✅ KNN model trained and saved as 'knn_model.pkl'")