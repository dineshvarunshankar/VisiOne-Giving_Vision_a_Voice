import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2


DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_PATH = 'checkpoints/depth_anything_v2_metric_hypersim_vits.pth' #model
ENCODER = 'vits'  # to be matched with modelsize
MAX_DEPTH = 5.0 #20 - hypersim(indoor), #80 - vkitti(outdoor)


# Model config
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Load model
model = DepthAnythingV2(**{**model_configs[ENCODER],'max_depth': MAX_DEPTH} )
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model = model.to(DEVICE).eval()

print(f"Depth Anything V2 loaded on {DEVICE.upper()}")
print("Press 'q' to quit.")

# Open webcam (0 = built-in, 1 = external USB)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    metric_depth = model.infer_image(frame)

    closest_distance_m = metric_depth.min()
    
    cv2.putText(frame, f"Closest: {closest_distance_m:.2f} m", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    depth_display = cv2.normalize(metric_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply a color map
    depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)

    # Combine and show
    combined = np.hstack([frame, depth_color])
    cv2.imshow('Visione Depth Test | RGB + Depth (Press Q to quit)', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test complete!")