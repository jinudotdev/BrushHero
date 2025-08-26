import mediapipe as mp
from camera_utils import CameraUtils

class FaceLogicProcessor:

    def __init__(self, painter, config):
        self.painter = painter
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.tesselation = self.mp_face_mesh.FACEMESH_TESSELATION

    def process_face_logic(self, frame_bgr, mode="full"):
        frame_rgb = CameraUtils.convert_bgr_to_rgb(frame_bgr)
        results = self.face_mesh.process(frame_rgb)
        face_landmarks_list = results.multi_face_landmarks if results.multi_face_landmarks else []

        for face_landmarks in face_landmarks_list:
            frame_bgr = self.painter.draw_face(
                frame_bgr,
                face_landmarks,
                self.tesselation,
                mode=mode
            )

        return frame_bgr
