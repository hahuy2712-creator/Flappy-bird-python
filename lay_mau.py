import cv2 as cv 
import os
user_name = input("Enter user name: ")
save_path = f"dataset/{user_name}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

face_model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    cv.imshow("Face Detection", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
cam.release()
cv.destroyAllWindows()