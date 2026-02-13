import librosa
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

DATASET_PATH = "dataset"

emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}

def extract_feature(file):
    audio, sr = librosa.load(file, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

features = []
labels = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotions[emotion_code]

            file_path = os.path.join(root, file)
            feature = extract_feature(file_path)

            features.append(feature)
            labels.append(emotion)

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl","wb"))
print("Model trained and saved")
