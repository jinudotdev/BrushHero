import cv2
import time

class CameraUtils:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
    def get_camera(self):
        return cv2.VideoCapture(self.camera_index)
    
    def convert_bgr_to_rgb(frame_bgr, debug=False):
        """Convert a BGR frame to RGB format for model input."""
        if debug:
            start = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if debug:
            print(f"[DEBUG] BGR→RGB conversion took {time.time() - start:.4f}s")
        return frame_rgb
