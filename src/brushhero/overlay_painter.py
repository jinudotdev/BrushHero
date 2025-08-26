import cv2

class OverlayPainter:
    # === [INIT] ===
    def __init__(self, debug=False):
        self.face_mesh_color = (255, 255, 255)   # White
        self.lip_color       = (0, 0, 255)       # Red
        self.palm_color      = (0, 165, 255)     # Orange
        self.finger_color    = (0, 255, 0)       # Green
        self.debug = debug

        self.LIPS_INDICES = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146
        ]

        self.LIP_CONNECTIONS = [
            (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
            (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
            (291, 375), (375, 321), (321, 405), (405, 314), (314, 17),
            (17, 84), (84, 181), (181, 91), (91, 146), (146, 61)
]
        self.PALM_CONNECTIONS = {frozenset(pair) for pair in [(0, 5), (0, 17), (5, 9), (9, 13), (13, 17)]}

    # === [DRAW ENTRYPOINT] ===
    def draw(self, frame, hand_data=None, face_data=None, line_data=None, tesselation=None, style="full"):
        if self.debug:
            cv2.putText(frame, f"Mode: {style}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.face_color, 1)
            print(f"[OverlayPainter] style={style}, hands={len(hand_data) if hand_data else 0}, faces={len(face_data) if face_data else 0}, lines={len(line_data) if line_data else 0}")

        if hand_data:
            frame = self.draw_hand(frame, hand_data, style)
        if face_data and tesselation:
            frame = self.draw_face(frame, face_data, tesselation, mode=style)
        if line_data:
            frame = self.draw_lines(frame, line_data)
        return frame

    # === [LINES] ===
    def draw_lines(self, frame, lines):
        for line in lines:
            pt1 = tuple(map(int, line["start"]))
            pt2 = tuple(map(int, line["end"]))
            color = line.get("color", (255, 0, 0))
            thickness = line.get("thickness", 2)
            cv2.line(frame, pt1, pt2, color, thickness)
        return frame

    # === [FACE] ===
    def draw_face(self, frame, face_landmarks, tesselation, mode="full"):
        def scale(pt): return int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])

        # Draw full face mesh
        # Face mesh lines
        for start_idx, end_idx in tesselation:
            x1, y1 = scale(face_landmarks.landmark[start_idx])
            x2, y2 = scale(face_landmarks.landmark[end_idx])
            cv2.line(frame, (x1, y1), (x2, y2), self.face_mesh_color, 1)

        # Lips overlay
        for start_idx, end_idx in self.LIP_CONNECTIONS:
            x1, y1 = scale(face_landmarks.landmark[start_idx])
            x2, y2 = scale(face_landmarks.landmark[end_idx])
            cv2.line(frame, (x1, y1), (x2, y2), self.lip_color, 2)
        return frame

    # === [HAND] ===
    def draw_hand(self, frame, hand_data, style="full"):
        for hand in hand_data:
            landmarks = hand.get('landmarks', [])
            connections = hand.get('connections', [])

            if style in ["full", "skeleton", "palm"] and landmarks and connections:
                for start_idx, end_idx in connections:
                    x1, y1 = landmarks[start_idx]
                    x2, y2 = landmarks[end_idx]
                    conn_key = frozenset((start_idx, end_idx))

                    color = self.palm_color if style == "palm" and conn_key in self.PALM_CONNECTIONS else self.finger_color
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            if style in ["full", "joints"] and landmarks:
                for x, y in landmarks:
                    cv2.circle(frame, (int(x), int(y)), 4, self.joint_color, 3)

        return frame
