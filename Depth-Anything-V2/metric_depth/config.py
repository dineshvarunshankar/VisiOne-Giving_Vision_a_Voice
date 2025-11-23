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

#Model Paths
DEPTH_MODEL_PATH = 'checkpoints/depth_anything_v2_metric_hypersim_vits.pth' #Path of the model
YOLO_MODEL = 'yolo11n-seg.pt'

#Model Parameters
ENCODER = 'vits' # 'vits' = Small, 'vitb' = Base, 'vitl' = Large
MAX_DEPTH = 1.0       # Max distance (in meters) for depth clipping - 20 for Hypersim(indoor), 80 for vkitti(outdoor)

#Application settings
CONF_THRESHOLD = 0.7
SAFETY_THRESHOLD = 1.5
INPUT_SIZE = (518, 518)

#Voice Model Params
AUDIO_COOLDOWN = 1.5
SPEAKER_WAV = 'reference.wav'