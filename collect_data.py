import cv2
import os

user_name = input("Enter user name: ")
save_path = f"dataset/{user_name}"

if not os.path.exists(save_path):
    os.makedirs(save_path)
#model nhận diện khuôn mặt của opencv
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

#thiết kế thuật toán nhận diện xem khuôn mặt ở đâu trong hình

cap = cv2.VideoCapture(0)
count = 0 #đếm số sample đã lấy

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #chuyển về dạng đen trắng để tiết kiệm tài nguyên tính toán
    denoise = cv2.GaussianBlur(gray, (5, 5), 0) # xử lý nhiễu 
    faces = face_cascade.detectMultiScale(denoise, 1.3, 5) #dùng ảnh đã khử nhiễu để đi nhận diện

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w] #lấy ảnh có chứa khuôn mặt 
        count += 1
        cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == 27 or count >= 100: #ESC == 27 hoặc đủ 100 ảnh 
        break

cap.release()
cv2.destroyAllWindows()