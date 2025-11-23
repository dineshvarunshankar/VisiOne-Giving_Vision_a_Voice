import torch
import numpy as np
import cv2
from depth_anything_3.api import DepthAnything3
from config import DEVICE, DEPTH_MODEL_REPO

class DepthEstimator:
    def __init__(self):
        print(f"[Depth] Loading {DEPTH_MODEL_REPO} on {DEVICE}...")
        # This forces the download into a folder named 'checkpoints' inside your project
        self.model = DepthAnything3.from_pretrained(DEPTH_MODEL_REPO, cache_dir="./checkpoints").to(DEVICE)
        self.model.eval()

    def get_full_depth_map(self, frame):
        
        original_h, original_w = frame.shape[:2]
        
        # 1. Convert to RGB (DA3 expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # 2. Native Inference
            # Passing a list [image] activates the internal batch processor
            prediction = self.model.inference([rgb_frame])
            
            # 3. Extract Result
            # .inference() handles the tensor conversion for us
            depth = prediction.depth[0] 
            
            if isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()

        # 4. Safety Resize (The Critical Fix)
        # DA3 often outputs 504x504 (multiples of 14). 
        # We resize back to 518x518 so it aligns with YOLO masks.
        if depth.shape[:2] != (original_h, original_w):
            depth = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        return depth

    def get_distance_in_mask(self, full_depth_map, mask):
        if mask is None or not mask.any():
            return None
        
        masked_vals = full_depth_map[mask]
        if len(masked_vals) == 0:
            return None
        
        # Use 5th percentile to avoid "black hole" noise (0.0 values)
        return np.percentile(masked_vals, 5)

    def visualize(self, depth_map):
        # Clip metric depth to 10m for better indoor contrast
        visual_depth = np.clip(depth_map, 0, 10.0)
        norm = (visual_depth - visual_depth.min()) / (visual_depth.max() - visual_depth.min() + 1e-6)
        vis = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)