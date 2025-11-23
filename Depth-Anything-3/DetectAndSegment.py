from ultralytics import YOLO
from config import DEVICE, YOLO_MODEL, CONF_THRESHOLD, INPUT_SIZE
import cv2
import torch
import numpy as np

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.model.to(DEVICE)

    def detect(self, frame):
        # Get current frame dimensions so we can resize masks to match
        frame_h, frame_w = frame.shape[:2]
        
        results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []

        if results.masks is not None:
            for mask_tensor, box, cls, conf in zip(results.masks.data, results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                # 1. Get raw mask (usually 640x640 or 160x160)
                raw_mask = mask_tensor.cpu().numpy().astype(np.uint8)
                
                # 2. Resize mask to match the actual input frame (518x518)
                # We use nearest neighbor interpolation to keep it binary (0 or 1)
                mask = cv2.resize(raw_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                label = self.model.names[int(cls.item())]
                bbox = [int(v) for v in box.tolist()]
                
                detections.append((label, mask, bbox, conf.item()))

        return detections
    
    def draw(self, frame, detections):
        annotated = frame.copy()
        for label, mask, (x1, y1, x2, y2), conf in detections:
            # Color: green for key obstacles, red for others
            color = (0, 255, 0) if label in ['person', 'chair', 'pole', 'bench'] else (0, 0, 255)
            
            # Overlay mask
            overlay = annotated.copy()
            
            # Safety check: Ensure mask matches frame size before applying
            if mask.shape[:2] == overlay.shape[:2]:
                overlay[mask] = color
                annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
            
            # Box + label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return annotated