import cv2
import numpy as np
import mediapipe as mp
from camera_utils import CameraUtils
from overlay_painter import OverlayPainter

class HandLogicProcessor:
    def __init__(self, painter: OverlayPainter, config):
        self.painter = painter
        self.config = config
        self.hand_detector = self._init_hand_detector()

    def _init_hand_detector(self):
        mp_hands = mp.solutions.hands
        return mp_hands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
    
    def process_hand_logic(self, frame_bgr, mode="palm"):
        frame_rgb = CameraUtils.convert_bgr_to_rgb(frame_bgr)
        hand_landmarks_list = self.detect_hand_landmarks(frame_rgb)

        hand_data = []
        for hand_landmarks in hand_landmarks_list:
            hand_data.append({
                "landmarks": [(lm.x * frame_bgr.shape[1], lm.y * frame_bgr.shape[0]) for lm in hand_landmarks.landmark], 
                "connections": self.get_hand_landmark_connections()
            })

        frame_bgr = self.painter.draw_hand(frame_bgr, hand_data, mode)
        return frame_bgr 

    def detect_hand_landmarks(self, frame_rgb):
        results = self.hand_detector.process(frame_rgb)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else []
    
    def get_hand_landmark_connections(self):
        return mp.solutions.hands.HAND_CONNECTIONS

    def classify_palm(self, hand_landmarks):
        palm_pts = [
            hand_landmarks.landmark[5],
            hand_landmarks.landmark[9],
            hand_landmarks.landmark[13],
            hand_landmarks.landmark[17],
            hand_landmarks.landmark[0],
        ]
        pts = np.array([[pt.x, pt.y] for pt in palm_pts])
        centroid = np.mean(pts, axis=0)
        area = cv2.contourArea(pts.astype(np.float32))
        spread = np.linalg.norm(pts[0] - pts[3])
        return {
            "pts": pts,
            "centroid": centroid,
            "area": area,
            "spread": spread
        }

    def get_render_payload(self, hand_landmarks, mode="palm"):
        if mode == "skeleton":
            return {"landmarks": hand_landmarks}
        elif mode == "palm":
            palm_data = self.classify_palm(hand_landmarks)
            return {"palm": palm_data}
        elif mode == "AR":
            palm_data = self.classify_palm(hand_landmarks)
            return {"AR_anchor": palm_data["pts"]}
        else:
            return {}
