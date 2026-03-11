import cv2
import os
import numpy as np

dataset_path = "dataset" #thư mục chứa ảnh khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create() #thuật toán nhận diện khuôn mặt của opencv

faces = []
labels = []
label_dict = {}
current_label = 0

for user in os.listdir(dataset_path):
    user_path = os.path.join(dataset_path, user)
    if not os.path.isdir(user_path):
        continue

    label_dict[current_label] = user

    for image_name in os.listdir(user_path):
        img_path = os.path.join(user_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(current_label)

    current_label += 1

recognizer.train(faces, np.array(labels))
recognizer.save("trainer/model.yml")

np.save("trainer/labels.npy", label_dict)

print("Training completed!")