import cv2
import math
import mediapipe as mp

def find_working_camera(max_index=5):
    for i in range(0,max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return 0  # fallback if none work

# Initialize camera
cam_index = find_working_camera()
cap = cv2.VideoCapture(cam_index)

# We're gonna use .drawing_utils to draw on the face. 
# we can define some details on how we want it drawn here
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def draw_landmark_overlay(
    frame,
    landmarks,
    connections,
    color,
    thickness
):
    h, w = frame.shape[:2] #extract frame height and width in pixels -> h, w
    # Draw connections
    for start_idx, end_idx in connections:
        start = landmarks.landmark[start_idx]
        end = landmarks.landmark[end_idx]

        # Convert normalized coordinates (0 to 1) to normal coordinates using image size
        pt1 = (int(start.x * w), int(start.y * h))
        pt2 = (int(end.x * w), int(end.y * h))

        cv2.line(frame, pt1, pt2, color, thickness)

def draw_distance_line(frame, pt1_norm, pt2_norm, threshold=0.12, label="Debug"):
    h, w = frame.shape[:2]
    pt1 = (int(pt1_norm[0] * w), int(pt1_norm[1] * h))
    pt2 = (int(pt2_norm[0] * w), int(pt2_norm[1] * h))

    print(f"Normalized: {pt1_norm} → {pt2_norm}")
    print(f"Pixels: {pt1} → {pt2}")

    dist = math.dist(pt1_norm, pt2_norm)
    color = (255, 0, 0) if dist < threshold else (0, 0, 255)

    # Always draw something
    cv2.line(frame, pt1, pt2, color, 2)
    cv2.putText(frame, f"{label}: {dist:.2f}", pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    cv2.circle(frame, mid, 5, (0, 255, 255), -1)
