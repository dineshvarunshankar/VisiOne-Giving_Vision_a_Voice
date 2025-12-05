import cv2
import numpy as np
from config import DISTANCE_WARN_NEAR

class Visualizer:
    def draw_composite(self, frame: np.ndarray, fused_detections: list, depth: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)
        mask_accumulator = np.zeros((h, w), dtype=bool)
        annotated = frame.copy()
        
        # Unpack 10 items
        for label, _, box, _, _, distance, valid_mask, velocity, direction, track_age in fused_detections:
            is_threat = "approaching" in direction or (distance is not None and distance < DISTANCE_WARN_NEAR)
            color = (0, 0, 255) if is_threat else (0, 255, 0)
            
            if valid_mask is not None and valid_mask.shape == (h, w):
                overlay[valid_mask] = color
                mask_accumulator |= valid_mask

            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            text_parts = [label]
            if distance: text_parts.append(f"{distance:.1f}m")
            
            if direction != "stationary":
                speed_ms = abs(velocity[2])
                if speed_ms > 0.2: text_parts.append(f"[{speed_ms:.1f}m/s]")
            

            text = " ".join(text_parts)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        if np.any(mask_accumulator):
            annotated[mask_accumulator] = cv2.addWeighted(annotated[mask_accumulator], 0.7, overlay[mask_accumulator], 0.3, 0)
            
        depth_vis = self._colorize_depth(depth)
        return np.vstack([annotated, depth_vis])

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        if depth is None: return np.zeros((10,10,3), dtype=np.uint8)
        norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-6)
        vis = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)