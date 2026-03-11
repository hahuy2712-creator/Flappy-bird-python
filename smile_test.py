import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

model_path = "face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(mp_image, timestamp)

    if result.face_blendshapes:

        blendshapes = result.face_blendshapes[0]

        smile_left = 0
        smile_right = 0

        for b in blendshapes:
            if b.category_name == "mouthSmileLeft":
                smile_left = b.score
            if b.category_name == "mouthSmileRight":
                smile_right = b.score

        smile_score = (smile_left + smile_right) / 2

        if smile_score > 0.5:
            cv2.putText(frame, "SMILE 😄",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        3)

    cv2.imshow("Smile Detection", frame)

    timestamp += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()