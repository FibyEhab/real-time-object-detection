from ultralytics import YOLO
import cv2
import random

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

model = YOLO("yolov8s.pt")

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, stream=True, verbose=False, conf=0.5)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            label = model.names[cls]

            text = f'{label} {conf:.2f}'
            cv2.putText(frame, text, (max(0, x1), max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Real Time", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
