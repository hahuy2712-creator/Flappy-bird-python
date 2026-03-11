import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
#load model
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
detector = vision.FaceDetector.create_from_options(options) #công cụ nhận diện

#load video
cam = cv2.VideoCapture(0)
timestamp = 0
while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break
    #chuyển hệ màu của opencv qua mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )  #chuẩn hoá ảnh đưa vào model 
   
    result = detector.detect_for_video(mp_image, timestamp)
    # Draw detections
    if result.detections:
        for detection in result.detections:

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            score = detection.categories[0].score

            cv2.putText(
                frame,
                f"Face {score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    cv2.imshow("Face Detector", frame)

    timestamp += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()