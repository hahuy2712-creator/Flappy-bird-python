import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model path
model_path = "hand_landmarker.task"

# Configure options
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

# Create detector
detector = vision.HandLandmarker.create_from_options(options)

# Define hand connections manually
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17)              # palm
]

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            points = []

            # convert normalized → pixel
            for lm in hand_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                points.append((x,y))

                cv2.circle(frame,(x,y),6,(0,255,0),-1)

            # # draw connections
            for start,end in HAND_CONNECTIONS:
                cv2.line(frame,points[start],points[end],(255,0,0),2)

    cv2.imshow("Hand Landmarks",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()