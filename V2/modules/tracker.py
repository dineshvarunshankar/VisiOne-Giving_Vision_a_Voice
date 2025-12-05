import numpy as np
import cv2
from collections import deque
from config import TRACK_BUFFER, VELOCITY_THRESHOLD, VELOCITY_SMOOTHING
from filterpy.kalman import KalmanFilter

class ObjectTrack:
    """Helper class to store history for a single object"""
    def __init__(self, kf, initial_age=1):
        self.kf = kf
        self.age = initial_age
        self.vel_history = deque(maxlen=VELOCITY_SMOOTHING)

class VelocityTracker:
    def __init__(self):
        self.tracks = {} 
        self.kalman_dim = 6 

    def update(self, detections, depth, dt):
        updated_detections = []
        
        # Stability check for time delta
        if dt <= 0.001: dt = 0.033
        if dt > 1.0: dt = 1.0 

        # 1. First Pass: Update Kalman Filters & Calculate Raw Velocities
        current_frame_velocities = []
        
        for item in detections:
            # Safe unpack (Expects 5 items from detector)
            label, mask, box, conf, track_id = item

            if mask is None or track_id == -1:
                 continue

            # Median Depth Calculation
            valid_mask = mask & (depth > 0)
            if not np.any(valid_mask):
                continue
            
            M = cv2.moments(valid_mask.astype(np.uint8))
            if M["m00"] == 0: 
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            z_vals = depth[valid_mask]
            centroid_z = np.median(z_vals)
            centroid_3d = np.array([cx, cy, centroid_z], dtype=np.float32)

            # Kalman Update
            if track_id not in self.tracks:
                kf = KalmanFilter(dim_x=6, dim_z=3)
                kf.F = np.eye(6) 
                kf.H = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]])
                kf.P *= 10.0
                kf.R *= 0.1 
                kf.Q *= 0.01 
                kf.x[:3] = centroid_3d.reshape((3, 1))
                self.tracks[track_id] = ObjectTrack(kf)
                
                # New track has 0 velocity initially
                current_frame_velocities.append(0.0)
            else:
                track = self.tracks[track_id]
                track.age += 1
                
                kf = track.kf
                kf.F[0, 3] = dt; kf.F[1, 4] = dt; kf.F[2, 5] = dt
                kf.predict()
                kf.update(centroid_3d)
                
                # Get Raw Z Velocity (Negative = Coming closer)
                # Convert array [x] to float scalar x
                raw_vz = float(kf.x[5]) 
                current_frame_velocities.append(raw_vz)

        # 2. Calculate Ego-Motion (Self Movement)
        ego_velocity_z = 0.0
        if len(current_frame_velocities) > 0:
            ego_velocity_z = np.median(current_frame_velocities)
            
            # Threshold to ignore noise
            if abs(ego_velocity_z) < 0.1: 
                ego_velocity_z = 0.0

        # 3. Second Pass: Adjust Velocities and Assign Directions
        for item in detections:
            label, mask, box, conf, track_id = item
            
            # Handle cases where tracking wasn't possible
            if mask is None or track_id == -1 or track_id not in self.tracks:
                 updated_detections.append((label, mask, box, conf, track_id, np.zeros(3), "stationary", 0))
                 continue

            track = self.tracks[track_id]
            
            # Raw Kalman Velocity
            raw_vel_3d = track.kf.x[3:6].flatten()
            
            # COMPENSATION STEP
            # Corrected Vz = Measured Vz - Ego Vz
            corrected_vz = float(raw_vel_3d[2]) - ego_velocity_z
            
            # Update the smoothed history with the CORRECTED value
            track.vel_history.append(corrected_vz)
            
            # Final Smooth Velocity
            final_vz = np.mean(track.vel_history)
            final_velocity = np.array([raw_vel_3d[0], raw_vel_3d[1], final_vz])

            # Direction Logic
            # Using VZ (forward/back speed) primarily for safety
            speed_abs = abs(final_vz)
            
            if speed_abs < VELOCITY_THRESHOLD:
                direction = "stationary"
            elif final_vz < -VELOCITY_THRESHOLD: 
                direction = "approaching" 
            elif final_vz > VELOCITY_THRESHOLD:
                direction = "receding"
            else:
                direction = "moving"
            
            updated_detections.append((label, mask, box, conf, track_id, final_velocity, direction, track.age))
        
        # Cleanup dead tracks
        active_ids = {d[4] for d in detections if d[4] != -1}
        self.tracks = {tid: t for tid, t in self.tracks.items() if tid in active_ids}
        
        return updated_detections