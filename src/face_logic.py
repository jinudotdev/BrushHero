import mediapipe as mp
import cv2
from overlay_painter import find_working_camera, draw_landmark_overlay

# Initialize camera
cam_index = find_working_camera()
cap = cv2.VideoCapture(cam_index)

# we imported mediapipe, now we're going to initialize a face mesh tool
mp_face_mesh = mp.solutions.face_mesh
# facemesh() starts the face landmark detector
# static_image_mods = False means it expects a video stream
# detection confidence is normally 0.5, we raised it so that it'll track
# a BIG mouth opening
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
# FACEMESH_TESSELATION defines the 468 landmark points 
# like eyes, nose, lips, etc.
# output for the coordinates are normalized (between 0 and 1) so it'll
# have to be multiplied by screen size for coordinates
FACEMESH_TESSELATION = mp.solutions.face_mesh.FACEMESH_TESSELATION

#our main function
def draw_face_overlay(frame):
    multi_face_landmarks = detect_face_landmarks(frame) # landmark function is called
    # Draw facial landmarks
    for face_landmarks in multi_face_landmarks: 
        draw_landmark_overlay( 
            frame,
            face_landmarks,
            FACEMESH_TESSELATION,
            color=(255, 255, 255),
            thickness=1,
        )

# Detect face landmarks from frame
def detect_face_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #BGR to RGB
    results = face_mesh.process(frame_rgb)
    return results.multi_face_landmarks if results.multi_face_landmarks else []