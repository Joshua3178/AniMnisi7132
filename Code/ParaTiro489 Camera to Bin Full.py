import cv2
from ultralytics import YOLOWorld

MODEL_PATH = r"E:\Life\School\Utar\Degree\Y4\S1\FYP 1\Code\VLM V3\yolov8x-worldv2.pt"

model = YOLOWorld(MODEL_PATH)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO-World Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()