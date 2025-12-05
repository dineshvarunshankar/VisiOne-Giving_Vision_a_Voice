import torch
import os

# ====================== HARDWARE ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.getcwd()

# ====================== MODELS ======================
YOLO_MODEL_PATH = "models/yolov8m-worldv2.pt" 
DEPTH_MODEL_VARIANT = "models/DA3"

# ====================== AUDIO ======================
MIC_DEVICE_INDEX = 1  
# TTS ENGINE SELECTOR
# Options: "ai" (Coqui XTTS - Realistic but slower) 
#          "system" (pyttsx3 - Robotic but instant)
TTS_ENGINE = "ai"
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE = "en"
TTS_SPEAKER_WAV = "models/female_01.wav"
TTS_BASE_RATE = 150.0
TTS_RATE = 160
WAKE_WORD = "hello" 
LISTENING_TIMEOUT = 3.0      

# ====================== SAFETY LOGIC ======================
# Percentage of screen width considered "Center". 0.33 means middle 33%.
CENTER_ZONE_RATIO = 0.20
DISTANCE_WARN_NEAR = 0.5    # Meters
VELOCITY_THRESHOLD = 0.5    # m/s
SPEED_ALERT_THRESHOLD = 5.0 # m/s (Urgent interrupt threshold)

MIN_TRACK_AGE = 15          # Frames (~0.5s) an object must exist before we warn about it
VELOCITY_SMOOTHING = 10      # Average speed over last 5 frames to fix "hand wave" spikes

COOLDOWN_SPEED   = 3.0      # For fast objects
COOLDOWN_GENERAL = 5.0      # For normal objects (Chair, Person) 



# ====================== TRACKING ======================
TRACK_BUFFER = 30           
TRACK_MATCH_THRESH = 0.8    

# ====================== TUNING ======================
INPUT_SIZE = (640, 384) 
DEPTH_SCALE = 0.8
PERCENTILE_CLOSEST = 5 
BACKGROUND_CUTOFF_RATIO = 1.20 

# ====================== VOCABULARY ======================
SAFETY_VOCABULARY = [
    # CRITICAL THREATS 
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "train", 
    "rider", "police car", "ambulance",
    
    # NAVIGATION HAZARDS 
    "stairs", "curb", "pothole", "pole", "tree", "traffic light", "stop sign", 
    "fire hydrant", "wet floor sign", "construction barrier", "fence",
    
    # INDOOR OBSTACLES 
    "chair", "table", "couch", "bed", "desk", "cabinet", "door", "doorway", 
    "window", "trash can", "box", "luggage", "backpack", "umbrella", 
    "shoe", "toy", "wire", "glass door",
    
    # INFRASTRUCTURE 
    "elevator", "escalator", "bench", "fountain", "atm", "vending machine",
    "counter", "sink", "toilet", "mirror",
    
    # COMMON OBJECTS (That you might knock over) 
    "cup", "bottle", "plate", "laptop", "monitor", "tv", "keyboard", 
    "phone", "keys", "book", "bag", "lamp", "plant"
]

# ====================== DEBUG ======================
SHOW_VISUALIZATION = True
WAIT_KEY_DELAY = 1
WINDOW_TITLE = "Visione | RGB + Depth"
SAVE_VIDEO = False
LOG_TO_FILE = False