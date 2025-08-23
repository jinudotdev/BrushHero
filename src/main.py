import cv2
from hand_logic import draw_hand_overlay
from face_logic import draw_face_overlay
#we're running a loop to find a working camera, the logic is under webcam_runner
from overlay_painter import find_working_camera

# Your camera setup and loop, function from webcam_runner
cam_index = find_working_camera()
print(f"Using camera index: {cam_index}")

cap = cv2.VideoCapture(cam_index)

# let's loop now
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #let's run the hand tracking
        draw_hand_overlay(frame)

        # let's run the face tracking
        draw_face_overlay(frame)

        cv2.imshow("BrushHero Overlay", frame)  # ← This is what opens the window

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # Esc or 'q'
            break

except KeyboardInterrupt:
    print("Interrupted by user")

cap.release()
cv2.destroyAllWindows()