import torch

# Auto-select best device
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
    print("MPS (Apple Silicon) detected")
else:
    DEVICE = 'cpu'
    print("CPU fallback")

# --- Model Paths ---
# DA3 uses the HuggingFace ID string (it downloads automatically)
# Or you can point to a local .safetensors file if you downloaded it manually
DEPTH_MODEL_PATH = 'depth-anything/DA3Metric-Large' 

YOLO_MODEL = 'yolo11n-seg.pt'

# --- Application settings ---
# DA3 handles internal resizing, but keeping a consistent input size is good for speed
INPUT_SIZE = (518, 518) 
CONF_THRESHOLD = 0.7
SAFETY_THRESHOLD = 2.0 # Updated: DA3 metric is very accurate, you might want a slightly larger buffer

# Voice Model Params
AUDIO_COOLDOWN = 1.5
SPEAKER_WAV = 'reference.wav'