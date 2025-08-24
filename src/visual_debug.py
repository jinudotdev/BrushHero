# visual_debug.py

import cv2

def draw_proximity_debug(frame, p1, p2, threshold=0.1, label=""):
    dist = _distance(p1, p2)
    color = (0, 255, 0) if dist < threshold else (0, 0, 255)
    cv2.line(frame, _to_px(p1), _to_px(p2), color, 2)
    if label:
        cv2.putText(frame, f"{label}: {dist:.2f}", _to_px(p1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_point_debug(frame, point, label=""):
    cv2.circle(frame, _to_px(point), 4, (255, 255, 0), -1)
    if label:
        cv2.putText(frame, label, _to_px(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def _distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def _to_px(norm_point, frame_shape=(640, 480)):
    x = int(norm_point[0] * frame_shape[0])
    y = int(norm_point[1] * frame_shape[1])
    return (x, y)
