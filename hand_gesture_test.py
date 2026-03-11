# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2, 
                                          running_mode=vision.RunningMode.VIDEO)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #chuyẻn hệ màu của opencv qua mediapipe

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    #chuẩn hoá ảnh đưa vào model 
    result = recognizer.recognize_for_video(mp_image, timestamp)

    if result.gestures:
        gesture = result.gestures[0][0].category_name
        score = result.gestures[0][0].score

        if gesture == "Thumb_Up":
            cv2.putText(frame, f"Thumbs Up 👍 ({score:.2f})",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 3)
        if gesture == "ILoveYou":
            cv2.putText(frame, f"I Love You 🤟 ({score:.2f})",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 5)

    cv2.imshow("Gesture", frame)

    timestamp += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

