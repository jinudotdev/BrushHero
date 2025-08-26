import cv2
import time
from AVFoundation import AVCaptureDevice

def get_camera_names():
    devices = AVCaptureDevice.devicesWithMediaType_("vide")
    return [str(device.localizedName()) for device in devices]

def scan_cameras(max_index=5, preview_duration=3):
    camera_names = get_camera_names()
    print("Detected camera names:")
    for idx, name in enumerate(camera_names):
        print(f"  [{idx}] {name}")

    for i in range(max_index):
        print(f"\nTrying camera index {i}...")
        cap = cv2.VideoCapture(i)

        if cap.isOpened():
            print(f"Camera {i} opened successfully. Previewing for {preview_duration} seconds...")
            start_time = time.time()

            while time.time() - start_time < preview_duration:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(f"Camera {i}", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Preview interrupted by user.")
                        break
                else:
                    print("Failed to read frame.")
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(f"Camera {i} closed.")
        else:
            print(f"Camera {i} failed to open.")

scan_cameras()
