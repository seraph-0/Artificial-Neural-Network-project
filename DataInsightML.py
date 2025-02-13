!unzip/content/drive/MyDrive/daily+and+sports+activities.zip -d /content/

!pip install stumpy

import stumpy
import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic time series or load your dataset
time_series = np.sin(np.linspace(0, 50, 1000))  # Example sine wave data

# Define subsequence length
m = 50

# Compute the matrix profile
mp = stumpy.stump(time_series, m)

# Plot the original time series and its matrix profile
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_series, label="Time Series")
plt.title("Original Time Series")
plt.subplot(2, 1, 2)
plt.plot(mp[:, 0], label="Matrix Profile", color='orange')
plt.title("Matrix Profile")
plt.tight_layout()
plt.show()

# Identify motifs and discords
motif_idx = np.argmin(mp[:, 0])  # Motif location
discord_idx = np.argmax(mp[:, 0])  # Discord location
print(f"Motif starts at index: {motif_idx}")
print(f"Discord starts at index: {discord_idx}")

import os
import numpy as np
import stumpy
import matplotlib.pyplot as plt

base_path = "/content/data"

for activity in os.listdir(base_path):
    activity_path = os.path.join(base_path, activity)

    for subject in os.listdir(activity_path):
        subject_path = os.path.join(activity_path, subject)

        for segment_file in os.listdir(subject_path):
            segment_path = os.path.join(subject_path, segment_file)
            data = np.loadtxt(segment_path, delimiter=",")
            torso_acc = data[:, 0:3]
            flat_torso_acc = torso_acc.flatten()
            m = 25
            mp = stumpy.stump(flat_torso_acc, m)
            motif_idx = np.argmin(mp[:, 0])
            discord_idx = np.argmax(mp[:, 0])
            # plt.figure(figsize=(12, 6))
            # plt.subplot(2, 1, 1)
            # plt.plot(flat_torso_acc, label="Torso Accelerometer (X, Y, Z)")
            # plt.title("Time Series Data")
            # plt.subplot(2, 1, 2)
            # plt.plot(mp[:, 0], label="Matrix Profile", color='orange')
            # plt.title("Matrix Profile")
            # plt.tight_layout()
            # plt.show()
            print(f"Motif starts at index: {motif_idx}")
            print(f"Discord starts at index: {discord_idx}")


import os
import numpy as np
import pandas as pd
import stumpy

base_path = "/content/data"
m = 25
feature_vectors = []
labels = []

for activity in os.listdir(base_path):
    activity_path = os.path.join(base_path, activity)
    activity_label = activity

    for subject in os.listdir(activity_path):
        subject_path = os.path.join(activity_path, subject)

        for segment_file in os.listdir(subject_path):
            segment_path = os.path.join(subject_path, segment_file)
            data = np.loadtxt(segment_path, delimiter=",")
            flat_data = data.flatten()
            mp = stumpy.stump(flat_data, m)
            mp_vector = mp[:, 0]
            feature_vectors.append(mp_vector)
            labels.append(activity_label)

features_df = pd.DataFrame(feature_vectors)
features_df['Label'] = labels
features_df.to_csv("matrix_profile_features.csv", index=False)
print("Feature extraction complete. Feature vectors saved to 'matrix_profile_features.csv'.")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("matrix_profile_features.csv")

X = data.drop(columns=["Label"])
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")


