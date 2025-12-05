import os, contextlib, time, cv2
import numpy as np
import torch.multiprocessing as mp
os.environ["NUMEXPR_MAX_THREADS"] = "16"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from modules.detector import OpenVocabDetector
from modules.depth import DepthEstimator
from modules.tracker import VelocityTracker
from modules.fusion import FusionEngine
from modules.safety_engine import SafetyEngine
from modules.tts import VoiceGuide
from modules.whisper_listener import QueryListener
from utils.visualizer import Visualizer
from utils.logger import Logger
from config import INPUT_SIZE, SHOW_VISUALIZATION, SAFETY_VOCABULARY, WAIT_KEY_DELAY, WINDOW_TITLE

def run_visione():
    logger = Logger()
    
    # Initialize Core Systems
    # We start RealSense here inside the silence block to suppress SDK logs
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            detector = OpenVocabDetector()
            depth_est = DepthEstimator()
            depth_est.start() 
    
    tracker = VelocityTracker()
    fusion = FusionEngine()
    safety = SafetyEngine()
    vis = Visualizer()
    guide = VoiceGuide() 
    listener = QueryListener(enabled=True)
    
    listener.start()
 
    
    last_time = time.time()
    logger.log("Visione Online.")

    # Main Loop
    while True:
        # Fetch frame from RealSense wrapper
        frame = depth_est.get_live_frame()
        if frame is None: 
            # Wait a brief moment if frame drop occurs
            time.sleep(0.01)
            continue
        
        dt = time.time() - last_time
        last_time = time.time()
        
        # Resize for processing (Standard Logic)
        frame = cv2.resize(frame, INPUT_SIZE)
        
        # 1. Query
        query = listener.get_query()
        if query: 
            logger.log(f"Query: {query}", "USER")
            clean_q = query.replace("find", "").replace("where is", "").strip()
            if len(clean_q) > 1: query = clean_q

        # 2. Detect
        prompts = SAFETY_VOCABULARY.copy()
        if query: prompts.append(query)
        
        detections = detector.detect(frame, prompts)
        
        # This returns the synced depth we captured in get_live_frame
        depth, _ = depth_est.estimate(frame)
        
        updated = tracker.update(detections, depth, dt)
        fused = fusion.fuse(updated, depth)
        
        # 3. Safety
        text, should_interrupt, is_critical = safety.process(fused, depth, query)
        
        if text:
            level = "URGENT" if is_critical else "INFO"
            logger.log(f"Speak: {text}", level)
            guide.speak(text, interrupt=should_interrupt)
        
        if SHOW_VISUALIZATION:
            cv2.imshow(WINDOW_TITLE, vis.draw_composite(frame, fused, depth))
        
        if cv2.waitKey(WAIT_KEY_DELAY) & 0xFF == ord('q'):
            break

    # Cleanup
    depth_est.stop()
    listener.stop()
    cv2.destroyAllWindows()
    logger.log("Visione Offline.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_visione()