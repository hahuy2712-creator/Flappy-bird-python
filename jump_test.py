import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

model_path = "pose_landmarker_heavy.task"

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

prev_hip_y = None
timestamp = 0

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    result = pose_landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.pose_landmarks:

        lm = result.pose_landmarks[0]

        left_hip = lm[23]
        right_hip = lm[24]

        hip_y = (left_hip.y + right_hip.y) / 2

        if prev_hip_y is not None:

            if prev_hip_y - hip_y > 0.05:
                cv2.putText(frame,"JUMP",(50,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,(0,255,0),4)

        prev_hip_y = hip_y

    cv2.imshow("Jump Detection",frame)

    if cv2.waitKey(1)==27:
        break