import mediapipe as mp
import cv2
from overlay_painter import find_working_camera, draw_landmark_overlay

# Initialize camera
cam_index = find_working_camera()
cap = cv2.VideoCapture(cam_index)

# this is mediapipe's tool for handling hands (no pun intended). Let's bring it in
mp_hands = mp.solutions.hands
# let's customize our hand instance. this is where we can finesse how we like it
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
# we're going to draw lines between our hand landmarks
mp_draw = mp.solutions.drawing_utils

# Determine grip status from landmarks
def draw_hand_overlay(frame):
    hand_landmarks_list = detect_landmarks(frame) #landmark function is called
    for hand_landmarks in hand_landmarks_list:
            draw_landmark_overlay(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                color=(0, 255, 0),
                thickness=2,
            )
    return hand_landmarks_list

# Detect hand landmarks from frame
def detect_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #BGR to RGB
    results = hands.process(frame_rgb)
    return results.multi_hand_landmarks if results.multi_hand_landmarks else []