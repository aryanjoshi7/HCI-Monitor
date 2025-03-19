import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    if len(pose_landmarks_list) == 0:
        return annotated_image
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 usually corresponds to the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Capture frames in a loop
while(True):
    # Read a frame
    ret, frame = cap.read()

    # Check if the frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Posture detection
    cv2.imshow('frame', frame)
    if frame is not None:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='/Users/aditkolli/Desktop/HCI-Monitor/HCI-Monitor/pose_landmarker_full.task'),
            running_mode=VisionRunningMode.IMAGE)

        with PoseLandmarker.create_from_options(options) as landmarker:
            print(frame)
            image_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            pose_landmarker_result = landmarker.detect(mp_image)
            print(pose_landmarker_result)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
            cv2.imshow('frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()