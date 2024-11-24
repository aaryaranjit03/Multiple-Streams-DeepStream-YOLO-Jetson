from ultralytics import YOLO
import cv2
model = YOLO("yolov8s.pt")
try:
    results = model.predict(source='rtsp://admin:abcd1234@192.168.1.64', show=True, conf=0.6)
    print(results)
except KeyboardInterrupt:
    print("\nProgram interrupted by user. Exiting...")
finally:
    cv2.destroyAllWindows()  # Ensures any open windows are closed