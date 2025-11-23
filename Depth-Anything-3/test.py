import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Prevents the Mac freeze

import torch
import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3

# 1. Setup Device
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load Model
# This model is for Metric Depth (meters), NOT for 3D GLB export.
model = DepthAnything3.from_pretrained("depth-anything/da3metric-large")
model = model.to(device=device)

# 3. Prepare Images
# IMPORTANT: Ensure these files actually exist in your folder!
images = ["image1.jpg", "image2.jpg"] 

# Check if images exist to avoid "NoneType" errors from cv2.imread later
valid_images = [img for img in images if os.path.exists(img)]
if not valid_images:
    print("❌ Error: No images found. Please check your filenames.")
else:
    # 4. Run Inference (CLEAN VERSION)
    # We removed 'export_format' to stop the crash.
    prediction = model.inference(valid_images)

    # 5. Access & Save Results
    for i, img_path in enumerate(valid_images):
        # Get the raw metric depth (in meters)
        depth = prediction.depth[i]
        
        # Normalize it to 0-255 so we can see it as a PNG
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_vis = depth_vis.astype(np.uint8)
        
        # Colorize for better visibility
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        
        # Save output
        output_name = f"result_{i}.png"
        cv2.imwrite(output_name, depth_vis)
        print(f"✅ Processed {img_path} -> Saved to {output_name}")

    # Verify shapes
    print(f"Depth Map Shape: {prediction.depth.shape}")