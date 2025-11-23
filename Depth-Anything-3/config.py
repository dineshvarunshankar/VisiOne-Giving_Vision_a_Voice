import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

_original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = safe_load


if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"[Config] Running on: {DEVICE.upper()}")

# Model & App Settings 
DEPTH_MODEL_REPO = 'depth-anything/DA3Metric-Large'
YOLO_MODEL = 'yolo11n-seg.pt'

INPUT_SIZE = (518, 518)
CONF_THRESHOLD = 0.6
SAFETY_THRESHOLD = 2.0    # meters
AUDIO_COOLDOWN = 2.0      # seconds
SPEAKER_WAV = 'reference.wav'