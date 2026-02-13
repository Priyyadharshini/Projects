from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

matrix = pickle.load(open("matrix.pkl","rb"))
similarity = pickle.load(open("similarity.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():
    recs = []

    if request.method == "POST":
        movie = request.form["movie"]

        if movie in matrix.index:
            idx = matrix.index.get_loc(movie)
            scores = list(enumerate(similarity[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

            for i in scores:
                recs.append(matrix.index[i[0]])

    return render_template("index.html", recs=recs)

if __name__ == "__main__":
    app.run(debug=True)
