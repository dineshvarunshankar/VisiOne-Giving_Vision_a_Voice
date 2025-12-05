import numpy as np
import time
from config import (
    DISTANCE_WARN_NEAR, SPEED_ALERT_THRESHOLD, 
    COOLDOWN_SPEED, COOLDOWN_GENERAL, MIN_TRACK_AGE,
    INPUT_SIZE, CENTER_ZONE_RATIO
)

class SafetyEngine:
    def __init__(self):
        # Speed Warning History
        self.last_spd_id = None
        self.last_spd_time = 0
        
        # General Obstacle History
        self.warn_history = {}
        
        # Memory Management
        self.last_cleanup_time = time.time()

    def process(self, fused_detections: list, depth_map: np.ndarray, query: str or None) -> tuple:
        """
        Returns: (text, should_interrupt, is_critical_speed)
        """
        current_time = time.time()
        
        # MEMORY CLEANUP (Runs every 30 seconds) 
        if current_time - self.last_cleanup_time > 30.0:
            self.warn_history = {
                k: v for k, v in self.warn_history.items() 
                if current_time - v['time'] < 60.0
            }
            self.last_cleanup_time = current_time

        # 1. SPEED THREATS (URGENT) 
        for label, _, box, _, track_id, distance, _, velocity, direction, track_age in fused_detections:
            if track_age < MIN_TRACK_AGE: continue
            
            speed_ms = abs(velocity[2]) if velocity is not None else 0.0
            
            # Check Position ("Side" vs "Ahead")
            location = self._get_location_text(box)
            is_center_threat = (location == "ahead")
            
            # Logic: Warn if (Fast & Center) OR (Fast & Side & Very Close)
            if direction == "approaching" and speed_ms > SPEED_ALERT_THRESHOLD:
                if not is_center_threat and (distance is not None and distance > 1.0):
                    continue

                uid = track_id if track_id != -1 else label
                
                if uid == self.last_spd_id and (current_time - self.last_spd_time < COOLDOWN_SPEED):
                    continue
                
                self.last_spd_id = uid
                self.last_spd_time = current_time
                
                guidance = self._get_guidance(box)
                msg = f"Warning! {label} approaching fast! {guidance}"
                return msg, True, True

        # 2. GENERAL OBSTACLES (INFO)
        closest_dist = 999.0
        closest_det = None
        
        for det in fused_detections:
            if det[9] < MIN_TRACK_AGE: continue 
            dist = det[5] 
            if dist is not None and dist < DISTANCE_WARN_NEAR:
                if dist < closest_dist:
                    closest_dist = dist
                    closest_det = det

        if closest_det:
            label = closest_det[0]
            box = closest_det[2]
            dist = closest_det[5]
            track_id = closest_det[4]
            uid = track_id if track_id != -1 else label
            
            should_speak = False
            
            if uid not in self.warn_history:
                should_speak = True
            else:
                last_data = self.warn_history[uid]
                time_diff = current_time - last_data['time']
                dist_diff = abs(dist - last_data['dist'])
                
                if time_diff > COOLDOWN_GENERAL or dist_diff > 0.8:
                    should_speak = True
            
            if should_speak:
                self.warn_history[uid] = {'time': current_time, 'dist': dist}
                guidance = self._get_guidance(box)
                msg = f"{label} {closest_dist:.1f} meters. {guidance}"
                return msg, True, False
            else:
                return None, False, False

        # 3. UNCLASSIFIED (Unknown Objects)
        h, w = depth_map.shape
        
        # Cut off the bottom 15% (Ignore Floor)
        # Old: y_end = h 
        # New: y_end = int(h * 0.85)
        y_start = h // 3
        y_end = int(h * 0.85) 
        
        x_start, x_end = w // 3, 2 * w // 3
        
        center_zone = depth_map[y_start:y_end, x_start:x_end]
        known_mask = np.zeros_like(depth_map, dtype=bool)
        for det in fused_detections:
             if det[6] is not None: known_mask |= det[6]
        
        unknown_mask = (~known_mask)[y_start:y_end, x_start:x_end]
        danger_pixels = (center_zone < DISTANCE_WARN_NEAR) & (center_zone > 0.1) & unknown_mask
        
        # Increased sensitivity threshold 
        # This prevents thin wires or noise from triggering "STOP"
        if np.sum(danger_pixels) > (center_zone.size * 0.15):
            uid = "generic_obstacle"
            should_speak = False
            if uid not in self.warn_history:
                should_speak = True
            else:
                if current_time - self.warn_history[uid]['time'] > COOLDOWN_GENERAL:
                    should_speak = True

            if should_speak:
                self.warn_history[uid] = {'time': current_time, 'dist': 1.0}
                ys, xs = np.where(danger_pixels)
                if len(xs) > 0:
                    avg_x_global = np.mean(xs) + x_start
                    dummy_box = [avg_x_global - 50, 0, avg_x_global + 50, 0]
                    guidance = self._get_guidance(dummy_box)
                else:
                    guidance = "Stop."

                return f"Obstacle ahead. {guidance}", True, False

        # 4. QUERY (Lowest Priority)
        if query:
            return self._process_query(query, fused_detections)

        return None, False, False

    def _get_guidance(self, box):
        x1, _, x2, _ = box
        center_x = (x1 + x2) / 2
        w = INPUT_SIZE[0]
        
        safe_left = w * (0.5 - CENTER_ZONE_RATIO/2)
        safe_right = w * (0.5 + CENTER_ZONE_RATIO/2)
        
        if center_x < safe_left: return "Move Right."
        elif center_x > safe_right: return "Move Left."
        else: return "Stop."

    def _get_location_text(self, box):
        x1, _, x2, _ = box
        center_x = (x1 + x2) / 2
        w = INPUT_SIZE[0]
        
        safe_left = w * (0.5 - CENTER_ZONE_RATIO/2)
        safe_right = w * (0.5 + CENTER_ZONE_RATIO/2)
        
        if center_x < safe_left: return "on your left"
        elif center_x > safe_right: return "on your right"
        else: return "ahead"

    def _process_query(self, query, fused_detections):
        target = query.lower()
        best_match = None
        min_dist = 999.0
        best_box = None
        
        for label, _, box, _, _, distance, _, _, _, _ in fused_detections:
            if (target in label.lower()) or (label.lower() in target):
                if distance is not None and distance < min_dist:
                    min_dist = distance
                    best_match = label
                    best_box = box
                    
        if best_match:
            loc = self._get_location_text(best_box)
            return f"Found {best_match} {loc}, {min_dist:.1f} meters.", False, False
            
        return f"I don't see the {target} yet.", False, False