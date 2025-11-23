import torch
import numpy as np
import cv2
from depth_anything_v2.dpt import DepthAnythingV2
from config import DEVICE, DEPTH_MODEL_PATH, ENCODER, INPUT_SIZE

class DepthEstimator:
    def __init__(self):
        print(f"[Depth] Loading Depth Anything V2 ({ENCODER}) on {DEVICE}...")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        self.model = DepthAnythingV2(**model_configs[ENCODER])
        self.model.load_state_dict(torch.load(DEPTH_MODEL_PATH, map_location='cpu'))
        self.model = self.model.to(DEVICE).eval()

    def preprocess(self, frame): #Convert BGR frame to normalized RGB tensor (1, C, H, W)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        return tensor

    def get_full_depth_map(self, frame): #Returns the full (H, W) metric depth map in meters.
        tensor = self.preprocess(frame)
        with torch.no_grad():
            depth = self.model(tensor).squeeze(0).cpu().numpy()  # (H, W)
        return depth
    def get_distance_in_mask(self, full_depth_map, mask):
        #Takes the full depth map and a mask, returns the minimum metric distance within that mask.
      
        if mask is None or not mask.any():
            return None
        
        masked_vals = full_depth_map[mask]
        
        if len(masked_vals) == 0:
            return None
        distance_m = masked_vals.min()
        
        return distance_m

    def visualize(self, depth_map): #Return colorized depth map for display
        norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        vis = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)