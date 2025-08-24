import cv2
from hand_logic import draw_hand_overlay
from face_logic import draw_face_overlay
#we're running a loop to find a working camera, the logic is under webcam_runner
from overlay_painter import find_working_camera, draw_distance_line
#debug GUI
from visual_debug import draw_proximity_debug

# Your camera setup and loop, function from webcam_runner
cam_index = find_working_camera()
print(f"FROM MAIN: Using camera index: {cam_index}")

cap = cv2.VideoCapture(cam_index)

# let's loop now
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #let's run the hand tracking
        hand_landmarks_list = draw_hand_overlay(frame)
  
        # let's run the face tracking
        multi_face_landmarks= draw_face_overlay(frame)
        
        #debug 
        debug = True
        if debug and bool(multi_face_landmarks) and bool(hand_landmarks_list):
            wrist = hand_landmarks_list[0].landmark[0]
            mouth = multi_face_landmarks[0].landmark[13]
            print(f"Wrist: {wrist.x:.2f}, {wrist.y:.2f} | Mouth: {mouth.x:.2f}, {mouth.y:.2f}")
            print("WE ARE ABOUT TO JUMP OVER TO THE OTHER SIDE")
            draw_distance_line(frame, (wrist.x, wrist.y), (mouth.x, mouth.y), threshold=0.12, label="Wrist→Mouth")


        cv2.imshow("BrushHero Overlay", frame)  # ← This is what opens the window

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # Esc or 'q'
            break
        

except KeyboardInterrupt:
    print("Interrupted by user")


cap.release()
cv2.destroyAllWindows()