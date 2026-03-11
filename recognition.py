import cv2
import numpy as np
from noti import show_popup
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/model.yml")

label_dict = np.load("trainer/labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)
checked = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        id_, confidence = recognizer.predict(face_roi)

        if confidence < 90:
            name = label_dict[id_]
            color = (0,255,0)
            if name.lower() == "tri":
                cv2.putText(frame, "Mo khoa thanh cong!", (x, y + h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if not checked:
                    show_popup()
                    checked = True
        else:
            name = "Unknown"
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, name, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()