import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Define the path to the downloaded model
model_path = 'face_landmarker.task'

# 2. Configure the Face Landmarker options
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True, # Set to True if you want expression scores (e.g., smile, blink)
    num_faces=1
)

# 3. Initialize the Landmarker
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # MediaPipe Tasks API requires its own 'mp.Image' format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # 4. Perform the detection
        detection_result = landmarker.detect(mp_image)

        # 5. Draw the landmarks using pure OpenCV (Bypassing mp.solutions entirely)
        if detection_result.face_landmarks:
            height, width, _ = image.shape
            for face_landmarks in detection_result.face_landmarks:
                for landmark in face_landmarks:
                    # The API returns normalized coordinates (0.0 to 1.0). 
                    # We must multiply by the image dimensions to get actual pixels.
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    
                    # Draw a tiny green dot for each landmark
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow('MediaPipe Face Landmarker (Tasks API)', image)
        
        # Press 'ESC' to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()