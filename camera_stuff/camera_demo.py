import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import serial

TARGET_EYE_HEIGHT = 0.55

arduino = serial.Serial(port='/dev/cu.usbmodem101', baudrate=115200, timeout=1) # TODO: change port maybe

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/Users/aryanjoshi/Documents/EECS_classes/HCI-Monitor/pose_landmarker_full.task'),
    running_mode=VisionRunningMode.IMAGE)
landmarker = PoseLandmarker.create_from_options(options)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 usually corresponds to the default camera




# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def send_command(command):
    print("sent", command)
    arduino.write(f"{command}\n".encode())
    time.sleep(0.1)
    response = arduino.readline().decode().strip()
    print(f"Arduino Response: {response}")
def write_read(x): 
    print("write_read")
    arduino.write(bytes(x, 'utf-8')) 
    time.sleep(0.05) 
    data = arduino.readline()
    print(data) 

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

def is_bad_posture(landmarks):
    shoulder_depth = (landmarks[0][12].z + landmarks[0][11].z)/2
    face_depth = landmarks[0][1].z
    # print(shoulder_depth)
    # print(face_depth)
    # print()
    return False




def initial_setup():
    # Adjust monitor so that it is at correct height for user
    ret, frame = cap.read()
    while frame is None or not ret:
        ret, frame = cap.read()
    image_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    pose_landmarker_result = landmarker.detect(mp_image)
    eye_level = pose_landmarker_result.pose_landmarks[0][2].y
    print("89", eye_level - TARGET_EYE_HEIGHT)

    while abs(eye_level - TARGET_EYE_HEIGHT) > 0.05:
        print(eye_level)
        if eye_level > TARGET_EYE_HEIGHT:
            # MOVE DOWN
            print("In this else case: ", str(eye_level))
            write_read(str(9))
        else:
            #MOVE UP
            print("In this test: ", str(eye_level))
            write_read(str(2))
        exit()
    
print("DOING INITIAL SETUP")
input()
initial_setup()
print("FINISHED INITIAL SETUP")
exit()
# Capture frames in a loop
bad_posture_duration = 0
BAD_POSTURE_DURATION_THRESHOLD = 10 # number of seconds user needs to have bad posture in order to prompt movement
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
        image_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        pose_landmarker_result = landmarker.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
        cv2.imshow('frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))   
        if len(pose_landmarker_result.pose_landmarks):
            print(pose_landmarker_result.pose_landmarks[0][2].y)
            if is_bad_posture(pose_landmarker_result.pose_landmarks):
                
                bad_posture_duration += 1
            else:
                bad_posture_duration = 0
            
            if bad_posture_duration > BAD_POSTURE_DURATION_THRESHOLD:
                # Move screen up
                pass

    time.sleep(0.25)
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()