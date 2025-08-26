# visual_debug.py
from src.brushhero.overlay_painter import find_working_camera, draw_distance_line

def debug_GUI(frame, debug, multi_face_landmarks, hand_landmarks_list):
    if debug and bool(multi_face_landmarks) and bool(hand_landmarks_list):
        wrist = hand_landmarks_list[0].landmark[0]
        mouth = multi_face_landmarks[0].landmark[13]
        print(f"Wrist: {wrist.x:.2f}, {wrist.y:.2f} | Mouth: {mouth.x:.2f}, {mouth.y:.2f}")
        draw_distance_line(frame, (wrist.x, wrist.y), (mouth.x, mouth.y), threshold=0.3, label="Wrist→Mouth")
 
