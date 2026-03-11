import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Define the path to the downloaded model
model_path = 'pose_landmarker_heavy.task'

# 2. Configure the Pose Landmarker options
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False, # Set to True if you want to remove the background
    num_poses=1
)

# Standard MediaPipe Pose Connections (Pairs of landmark indices to draw the bones)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

# 3. Initialize the Landmarker
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    
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

        # 4. Perform pose detection
        detection_result = landmarker.detect(mp_image)

        # 5. Draw the skeleton using pure OpenCV
        if detection_result.pose_landmarks:
            height, width, _ = image.shape
            for pose_landmarks in detection_result.pose_landmarks:
                
                # First, convert all normalized coordinates to exact pixel coordinates
                pixel_landmarks = []
                for landmark in pose_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    pixel_landmarks.append((x, y))
                    
                    # Draw the joints (red dots)
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                
                # Next, draw the bones (green lines) by connecting the joints
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    pt1 = pixel_landmarks[start_idx]
                    pt2 = pixel_landmarks[end_idx]
                    cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Tasks API - Pose', image)
        
        # Press 'ESC' to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()