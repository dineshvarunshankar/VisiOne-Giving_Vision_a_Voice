import numpy as np
from config import PERCENTILE_CLOSEST, BACKGROUND_CUTOFF_RATIO

class FusionEngine:
    def fuse(self, detections: list, depth: np.ndarray) -> list:
        fused_detections = []
        h_img, w_img = depth.shape[:2]
        
        # Unpack now includes 'track_age'
        for label, mask, box, conf_det, track_id, velocity, direction, track_age in detections:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            depth_patch = depth[y1:y2, x1:x2]
            valid_pixels = depth_patch[depth_patch > 0]
            
            if valid_pixels.size == 0:
                fused_detections.append((label, mask, box, conf_det, track_id, None, None, velocity, direction, track_age))
                continue

            # Metric Foreground Logic
            k = int(valid_pixels.size * (PERCENTILE_CLOSEST / 100.0))
            if k < valid_pixels.size:
                closest_val = np.partition(valid_pixels, k)[k]
            else:
                closest_val = np.min(valid_pixels)
            
            cutoff = closest_val * BACKGROUND_CUTOFF_RATIO
            local_mask = depth_patch <= cutoff
            
            foreground_depths = depth_patch[local_mask]
            distance = np.mean(foreground_depths) if foreground_depths.size > 0 else closest_val
            
            full_mask = np.zeros((h_img, w_img), dtype=bool)
            full_mask[y1:y2, x1:x2] = local_mask

            fused_detections.append((label, mask, box, conf_det, track_id, distance, full_mask, velocity, direction, track_age))
        
        return fused_detections