from ultralytics import YOLOWorld, YOLO
from config import DEVICE, YOLO_MODEL_PATH
import numpy as np
import cv2
import os

class OpenVocabDetector:
    def __init__(self):
       
        project_root = os.getcwd() 
        model_path = os.path.join(project_root, YOLO_MODEL_PATH) if not os.path.exists(YOLO_MODEL_PATH) else YOLO_MODEL_PATH

        self.is_world_model = "world" in YOLO_MODEL_PATH.lower()
        print(f"Loading Detector: {model_path} (World Mode: {self.is_world_model})...")
        
        # Load Model
        self.model = YOLOWorld(model_path) if self.is_world_model else YOLO(model_path)
        self.model.to(DEVICE)
        self.last_prompts = []

    def detect(self, frame: np.ndarray, prompts: list) -> list:
        # Optimization: Update classes only on change
        if self.is_world_model and sorted(prompts) != sorted(self.last_prompts):
            self.model.set_classes(prompts)
            self.last_prompts = prompts.copy()
        
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for i in range(len(results.boxes)):
                label = results.names[int(results.boxes.cls[i])]
                conf = float(results.boxes.conf[i])
                box = [int(v) for v in results.boxes.xyxy[i]]
                
                
                track_id = int(results.boxes.id[i]) if results.boxes.id is not None else -1
                
                # Mask Fallback
                if results.masks is not None:
                    mask = results.masks.data[i].cpu().numpy().astype(bool)
                    if mask.shape[:2] != frame.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask = np.zeros(frame.shape[:2], dtype=bool)
                    x1, y1, x2, y2 = box
                    # Clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    mask[y1:y2, x1:x2] = True
                
                detections.append((label, mask, box, conf, track_id))
        
        return detections