import cv2
import winsound
import threading

# Load cascade files
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier(
    "haarcascade_eye_tree_eyeglasses.xml")

# Start webcam (Windows stable mode)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

closed_frames = 0
threshold = 30   # frames before alert

def play_alarm():
    winsound.Beep(2500, 1000)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        # Correct logic
        if len(eyes) > 0:
            closed_frames = 0
        else:
            closed_frames += 1

        # Display counter
        cv2.putText(frame, f"Closed Frames: {closed_frames}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,255), 2)

        # Alert if threshold exceeded
        if closed_frames > threshold:
            cv2.putText(frame, "DROWSY ALERT!",
                        (100,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0,0,255), 3)

            threading.Thread(target=play_alarm).start()

        # Draw face box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
