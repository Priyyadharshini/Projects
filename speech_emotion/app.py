from flask import Flask, render_template, request
import pickle
import librosa
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

def extract_feature(file):
    audio, sr = librosa.load(file, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs.reshape(1,-1)

@app.route("/", methods=["GET","POST"])
def home():
    result = ""
    if request.method == "POST":
        file = request.files["file"]
        file.save("temp.wav")

        feature = extract_feature("temp.wav")
        pred = model.predict(feature)

        result = pred[0]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
