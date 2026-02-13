from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():
    result = ""

    if request.method == "POST":
        values = [float(x) for x in request.form.values()]
        data = np.array(values).reshape(1,-1)

        data = scaler.transform(data)
        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "Parkinson Detected"
        else:
            result = "Healthy"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
