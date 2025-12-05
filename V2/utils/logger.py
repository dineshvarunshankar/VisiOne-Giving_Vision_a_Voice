import datetime
import cv2
import numpy as np
from config import LOG_TO_FILE, SAVE_VIDEO, INPUT_SIZE

class Logger:
    def __init__(self):
        self.log_file = None
        self.video_writer = None
        
        if LOG_TO_FILE:
            self.log_file = open("visione_log.txt", "a")
        
        if SAVE_VIDEO:
            video_w, video_h = INPUT_SIZE[0], INPUT_SIZE[1] * 2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter("visione_output.mp4", fourcc, 10.0, (video_w, video_h))

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_str = f"[{timestamp}] [{level}] {message}"
        print(log_str)
        if self.log_file:
            self.log_file.write(log_str + "\n")
            self.log_file.flush()

    def log_detection(self, fused_detections: list):
        if not fused_detections:
            return
        
        summary = []
        for label, _, _, conf, _, distance, _, velocity, direction, age in fused_detections:
            dist_str = f"{distance:.1f}m" if distance else "None"
            dir_str = direction[:4].title() if direction != "stationary" else "Stat"
            summary.append(f"{label} ({dir_str}, {dist_str})")

    def save_frame(self, combined_visual: np.ndarray):
        if self.video_writer:
            self.video_writer.write(combined_visual)

    def close(self):
        if self.log_file: self.log_file.close()
        if self.video_writer: self.video_writer.release()