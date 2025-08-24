def is_hand_near_face(hand_landmarks, face_landmarks, threshold=0.28):
    """
    Returns True if hand is close enough to face.
    Assumes normalized coordinates (0–1 range).
    """
    if not hand_landmarks or not face_landmarks:
        return False

    wrist = hand_landmarks.get('wrist')
    mouth = face_landmarks.get('mouth_center') or face_landmarks.get('lips')

    if not wrist or not mouth:
        return False

    dx = wrist[0] - mouth[0]
    dy = wrist[1] - mouth[1]
    dist = (dx**2 + dy**2)**0.5

    return dist < threshold
