import pyrealsense2 as rs
import numpy as np
import cv2
from config import DEVICE

class DepthEstimator:
    def __init__(self):
        print("[Depth] Initializing Intel RealSense D455...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        try:
            self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        except Exception:
            print("[Depth Warning] High-Res failed. Falling back to 640x480 (Check USB Cable).")
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Align depth to color so pixels match exactly
        self.align = rs.align(rs.stream.color)
        
        # Filters to smooth the output
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        
        self.active = False
        self.latest_depth_map = None 
        self.depth_scale = 0.001 

    def start(self):
        """Starts the camera pipeline"""
        try:
            profile = self.pipeline.start(self.config)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.active = True
            print(f"[Depth] RealSense Active. Scale: {self.depth_scale}")
        except Exception as e:
            print(f"[Depth Critical] Failed to start pipeline: {e}")
            self.active = False

    def get_live_frame(self):
        """
        Fetches the latest synchronized Color frame from RealSense.
        Replaces cv2.VideoCapture.read()
        """
        if not self.active: return None
        
        try:
            # Wait for coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None
            
            # 1. Process Depth immediately and cache it
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            
            depth_data = np.asanyarray(depth_frame.get_data())
            self.latest_depth_map = depth_data * self.depth_scale
            
            # 2. Return Color image (numpy array)
            return np.asanyarray(color_frame.get_data())
            
        except Exception as e:
            print(f"[Depth Loop Error] {e}")
            return None

    def estimate(self, frame: np.ndarray) -> tuple:
        """
        Returns the depth map cached from the last get_live_frame() call.
        Resizes it to match the current frame (which main.py might have resized).
        """
        target_h, target_w = frame.shape[:2]
        
        if self.latest_depth_map is None:
            return np.zeros((target_h, target_w)), np.zeros((target_h, target_w))
            
        metric_depth = self.latest_depth_map
        
        # Resize to match the logic in main.py (INPUT_SIZE)
        # We use Nearest Neighbor to keep depth edges sharp
        if metric_depth.shape[:2] != (target_h, target_w):
            metric_depth = cv2.resize(metric_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
        # Confidence: RealSense returns 0 for invalid pixels
        conf = (metric_depth > 0).astype(np.float32)
        
        return metric_depth, conf

    def stop(self):
        if self.active:
            self.pipeline.stop()