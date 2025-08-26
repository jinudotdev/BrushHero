import cv2
from config import Config
from camera_utils import CameraUtils
from hand_logic import HandLogicProcessor
from face_logic import FaceLogicProcessor
from overlay_painter import OverlayPainter


cam_utils = CameraUtils()
cap = cv2.VideoCapture(1)  # Force FaceTime HD Camera
config = Config()
painter = OverlayPainter()
hand_processor = HandLogicProcessor(painter, config)
face_processor = FaceLogicProcessor(painter, config)

while True:
    ret, frame = cap.read()
    if frame is None or frame.size == 0:
        print("Empty frame received")
        continue
    if not ret:
        raise RuntimeError("Failed to read from selected camera.")

    frame = hand_processor.process_hand_logic(frame, mode="palm")
    frame = face_processor.process_face_logic(frame, mode="lips")

    cv2.imshow("Overlay", frame)

    if cv2.waitKey(1) in [27, ord('q')]:  # 27 = Escape, 'q' = quit
        break

cap.release()
cv2.destroyAllWindows()