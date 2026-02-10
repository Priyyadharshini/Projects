from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():
    prediction = ""
    if request.method == "POST":
        text = request.form["review"]
        vec = vectorizer.transform([text])
        pred = model.predict(vec)

        if pred[0] == 1:
            prediction = "Positive Review"
        else:
            prediction = "Negative Review"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
