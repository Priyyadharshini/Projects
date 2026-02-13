from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

kmeans = pickle.load(open("kmeans.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():
    segment = ""

    if request.method == "POST":
        income = float(request.form["income"])
        score = float(request.form["score"])

        data = scaler.transform([[income, score]])
        pred = kmeans.predict(data)

        segment = f"Customer belongs to Segment {pred[0]}"

    return render_template("index.html", segment=segment)

if __name__ == "__main__":
    app.run(debug=True)
