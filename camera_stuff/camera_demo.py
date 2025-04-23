import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import serial
import csv
import select
import sys
TARGET_EYE_HEIGHT = 0.5

arduino = serial.Serial(port='/dev/tty.usbmodem101', baudrate=9600, timeout=1) # TODO: change port maybe

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/Users/aditkolli/Desktop/HCI-Monitor/HCI-Monitor/pose_landmarker_full.task'),
    running_mode=VisionRunningMode.IMAGE)
landmarker = PoseLandmarker.create_from_options(options)

# Initialize the camera
cap = cv2.VideoCapture(1)  # 0 usually corresponds to the default camera




# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def send_command(command):
    arduino.write(f"{command}\n".encode())
    time.sleep(0.1)
    try:
        response = arduino.readline().decode().strip()
        print(f"Arduino Response: {response}")
    except Exception as E:
        print("No response from Arduino")

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

def detect_landmarks():
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    # time.sleep(1)
    while frame is None or not ret:
        ret, frame = cap.read()
    
    image_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    pose_landmarker_result = landmarker.detect(mp_image)
    if len(pose_landmarker_result.pose_landmarks):
        return pose_landmarker_result
    return detect_landmarks()


def initial_setup():
    # Adjust monitor so that it is at correct height for user
    pose_landmarker_result = detect_landmarks()
    eye_level = pose_landmarker_result.pose_landmarks[0][2].y

    while abs(eye_level - TARGET_EYE_HEIGHT) > 0.06:
        print(eye_level)
        if eye_level > TARGET_EYE_HEIGHT:
            # MOVE DOWN
            print("HEAD IS TOO LOW")
            send_command(2)
        else:
            # MOVE up
            print("HEAD IS TOO HIGH")
            send_command(3)
        time.sleep(1)
        pose_landmarker_result = detect_landmarks()
        eye_level = pose_landmarker_result.pose_landmarks[0][2].y
    # TARGET_EYE_HEIGHT = eye_level
    print("HEAD IS AT CORRECT HEIGHT")

print("DOING INITIAL SETUP")
initial_setup()
print("FINISHED INITIAL SETUP")
print("-----------------------------------")
print()
print()
# Capture frames in a loop
bad_posture_duration = 0
BAD_POSTURE_DURATION_THRESHOLD = 5 # number of seconds user needs to have bad posture in order to prompt movement
nudges = 0
total_bad_posture_duration = 0
total_duration = 0
TOLERANCE = 0.08
with open("data_aryan.csv", mode='w', newline='') as csvfile:
    fieldnames = ["time", "time in bad posture", "nudges", "tolerance", "duration threshold", "target height", "real height"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    mode = "POSTURE_DETECTION"
    while(True):
        # if total_duration % 5 == 0:
        #     mode = input()
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            user_input = sys.stdin.readline().strip()
            print("You typed:", user_input)
            if user_input[-1] == 'P':
                mode = "POSTURE_DETECTION"
            if user_input[-1] == 'H':
                mode = "HEIGHT_ADJUST"
                print("MODE CHANGED TO HEIGHT ADJUST")
            if user_input[-1] == 'N':
                mode = "NO_MOVE"

        pose_landmarker_result = detect_landmarks()
        eye_level = pose_landmarker_result.pose_landmarks[0][2].y
        if mode == "POSTURE_DETECTION":
            nudged = False
            print()
            print(eye_level)
            if abs(eye_level - TARGET_EYE_HEIGHT) >= TOLERANCE:
                print("BAD POSTURE")
                bad_posture_duration += 1
                total_bad_posture_duration += 1
            else:
                bad_posture_duration = 0
            if bad_posture_duration > BAD_POSTURE_DURATION_THRESHOLD:
                print("BAD POSTURE DETECTED; NUDGING")
                bad_posture_duration = 0
                send_command(5) # For vibrate
                nudges += 1
            total_duration += 1
            time.sleep(1)
            print("STATISTICS")
            print("-----------------------")
            print("total duration: ", total_duration)
            print("bad posture duration: ", total_bad_posture_duration)
            print("consecutive seconds in bad posture", bad_posture_duration)
            print("nudges: ", nudges)
            
            toPrint = {"time": total_duration, "time in bad posture": total_bad_posture_duration, "nudges": nudges, 
            "tolerance": TOLERANCE, "duration threshold": BAD_POSTURE_DURATION_THRESHOLD, 
            "target height": TARGET_EYE_HEIGHT, "real height": eye_level}
            writer.writerow(toPrint)
            if total_duration > 480:
                TOLERANCE = 0.2
                BAD_POSTURE_DURATION_THRESHOLD = 10
        if mode == "HEIGHT_ADJUST":
            print(eye_level)
            if eye_level > TARGET_EYE_HEIGHT + .03:
                # MOVE DOWN
                print("HEAD IS TOO LOW")
                send_command(2)
            if eye_level < TARGET_EYE_HEIGHT - .03:
                # MOVE up
                print("HEAD IS TOO HIGH")
                send_command(3)
            time.sleep(0.5)
        if mode == "NO_MOVE":
            print(eye_level)
# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()