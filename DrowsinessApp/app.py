from flask import Flask, render_template, Response
import cv2
import winsound
import threading

app = Flask(__name__)

# Load cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# Start webcam (Windows stable mode)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

closed_frames = 0
threshold = 30

def play_alarm():
    winsound.Beep(2500, 1000)

def generate_frames():
    global closed_frames

    while True:
        success, frame = camera.read()

        # Camera safety check
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            if len(eyes) > 0:
                closed_frames = 0
            else:
                closed_frames += 1

            if closed_frames > threshold:
                threading.Thread(target=play_alarm).start()
                cv2.putText(frame, "DROWSY ALERT!",
                            (100,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0,0,255), 3)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
