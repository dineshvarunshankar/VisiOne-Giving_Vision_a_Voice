from depth_anything_3.api import DepthAnything3
from config import DEVICE, DEPTH_MODEL_VARIANT, DEPTH_SCALE
import numpy as np
import os
import cv2
import torch
import contextlib 

class DepthEstimator:
    def __init__(self):
        project_root = os.getcwd()
        if os.path.exists(DEPTH_MODEL_VARIANT):
            model_path = DEPTH_MODEL_VARIANT
        else:
            model_path = os.path.join(project_root, DEPTH_MODEL_VARIANT)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Metric Depth Model not found at: {model_path}")

        print(f"Loading Metric DA3 from local: {model_path}...")
        try:
            self.model = DepthAnything3.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            print(f"[Depth Init Error] Failed to load local model: {e}")
            raise

        self.model.to(DEVICE).eval()

    def estimate(self, frame: np.ndarray) -> tuple:
        original_h, original_w = frame.shape[:2]

        try:
            # 1. BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Inference
            with open(os.devnull, 'w') as fnull:
                with contextlib.redirect_stdout(fnull):
                    prediction = self.model.inference([frame_rgb])
            
            if prediction.depth is not None:
                raw_depth = prediction.depth[0]
                
                # 3. Resize if needed
                if raw_depth.shape[:2] != (original_h, original_w):
                    raw_depth = cv2.resize(raw_depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                
                # 4. Direct Metric Output
                metric_depth = raw_depth * DEPTH_SCALE
                
                # 5. Handle Confidence
                if prediction.conf is not None:
                    conf = prediction.conf[0]
                    if conf.shape[:2] != (original_h, original_w):
                        conf = cv2.resize(conf, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                else:
                    conf = np.ones_like(metric_depth)

                return metric_depth, conf
            
            return np.zeros((original_h, original_w)), np.zeros((original_h, original_w))

        except Exception as e:
            print(f"[Depth Error] {e}")
            return np.zeros((original_h, original_w)), np.zeros((original_h, original_w))