from flask import Flask, render_template, request
from deepface import DeepFace
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET","POST"])
def home():
    result = ""
    img_path = ""

    if request.method == "POST":
        file = request.files["image"]
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(img_path)

        analysis = DeepFace.analyze(img_path, actions=['age','gender'])

        age = analysis[0]['age']
        gender = analysis[0]['dominant_gender']

        result = f"{gender}, Age approx: {age}"

    return render_template("index.html", result=result, img=img_path)

if __name__ == "__main__":
    app.run(debug=True)
