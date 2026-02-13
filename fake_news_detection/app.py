from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():
    prediction = ""
    if request.method == "POST":
        news = request.form["news"]
        vec = vectorizer.transform([news])
        pred = model.predict(vec)
        prediction = pred[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
