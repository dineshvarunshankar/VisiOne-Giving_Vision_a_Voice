import cv2
from ultralytics import YOLO
import torch
import numpy as np

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL = 'yolo11m-seg.pt'  # segmentation model
CONF = 0.5

# Load YOLO11n-seg
yolo = YOLO(MODEL)
yolo.to(DEVICE)

# Open webcam
cap = cv2.VideoCapture(0)
print("Running detection + segmentation | Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 384))
    results = yolo(frame, conf=CONF, verbose=False)[0]

    annotated = frame.copy()
    masks = []  # list of (label, mask)

    if results.masks is not None:
        for mask_tensor, box, cls, conf in zip(
            results.masks.data, results.boxes.xyxy, results.boxes.cls, results.boxes.conf
        ):
            mask = mask_tensor.cpu().numpy().astype(bool)
            cls_name = yolo.names[int(cls)]
            color = (0, 255, 0) if 'person' in cls_name else (255, 0, 0)

            
            # Draw mask using NumPy broadcasting for weighted add
            annotated[mask] = (annotated[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
            

            # Draw box + label
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            masks.append((cls_name, mask))

    cv2.imshow('YOLO11: Detection + Segmentation', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()