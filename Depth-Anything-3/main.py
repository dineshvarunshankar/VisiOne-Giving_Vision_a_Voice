import os
import cv2
import numpy as np
import time

# Import custom modules``
import config
from DetectAndSegment import ObjectDetector
from DepthEstimation import DepthEstimator
from VoiceModel import VoiceGuide

if config.DEVICE == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 1. Initialize Modules
print(f"Using device: {config.DEVICE.upper()}")

detector = ObjectDetector()
depth_est = DepthEstimator()
voice = VoiceGuide()  # Uses xTTS

# Initialize App State
#video_path = 'video1.mp4'
cap = cv2.VideoCapture(0)
last_spoken_time = 0.0
print("Visione running | Press Q to quit")

# 3. Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to standard size from config
    frame = cv2.resize(frame, config.INPUT_SIZE)
    h, w = frame.shape[:2]

    # Step 1: Detect + Segment 
    detections = detector.detect(frame)
    annotated_frame = detector.draw(frame, detections)

    # Step 2: Run Depth Model (ONCE per frame)
    full_depth_map = depth_est.get_full_depth_map(frame)

    # Step 3: Process Detections & Guide User
    spoken_this_frame = False # Prevents multiple audios in one frame
    
    for label, mask, bbox, conf in detections:
        # Get distance for this *specific* object
        distance = depth_est.get_distance_in_mask(full_depth_map, mask)
        
        if distance is None:
            continue

        # Draw distance text on the annotated frame
        x1, y1, _, _ = bbox
        cv2.putText(annotated_frame, f"{distance:.1f}m", (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Audio Guidance Logic
        # Speak only for the first obstacle under the safety threshold
        if distance < config.SAFETY_THRESHOLD and not spoken_this_frame:
            
            current_time = time.time()
            # Check if cooldown has passed
            if (current_time - last_spoken_time) > config.AUDIO_COOLDOWN:
                last_spoken_time = current_time  # Reset timer
                
                # Create the warning text
                warning_text = f"{label} ahead, {distance:.1f} meters"
                
                # Check position for directional guidance
                box_center_x = (bbox[0] + bbox[2]) / 2
                if box_center_x < w / 3:
                    warning_text += ", move right"
                elif box_center_x > w * 2 / 3:
                    warning_text += ", move left"
                else:
                    warning_text += ", move carefully"

                # Speak the warning
                voice.speak(warning_text)
                spoken_this_frame = True # Mark as spoken for this frame

    # Step 4: Visualize Full Depth Map
    # We already have full_depth_map, just visualize it
    depth_color = depth_est.visualize(full_depth_map)

    # Step 5: Display ---
    # Combine: Annotated RGB + Depth
    combined = np.vstack([annotated_frame, depth_color])
    cv2.imshow('Visione | Segmented + Depth + Voice (Q to quit)', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Visione stopped.")