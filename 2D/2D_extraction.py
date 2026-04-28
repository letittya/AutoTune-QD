import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import canny

# ── 1. Setup Paths (CLI or Fallback) ────────────────────────
# We use the same image you used for 1D to keep it a fair test
if len(sys.argv) > 1:
    input_path = sys.argv[1]
    img_basename = os.path.splitext(os.path.basename(input_path))[0]
    out_folder = os.path.join("testing_2D", "results", img_basename)
else:
    input_path = os.path.join("CSD_generated_images", "csd_clean.png")
    out_folder = "2D"

os.makedirs(out_folder, exist_ok=True)

if not os.path.exists(input_path):
    print(f"Cannot find {input_path}")
    sys.exit(1)

# ── 2. Load and Preprocess (Identical to 1D) ────────────────
img = plt.imread(input_path)

# Grayscale
if img.ndim == 3:
    img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
else:
    img_gray = img

# Smoothing (Fixed sigma=2.0 for consistency)
img_smoothed = gaussian_filter(img_gray, sigma=2.0)

# ── 3. Canny Edge Detection (The 2D Step) ───────────────────
# We convert to uint8 for standard Canny processing
img_uint8 = (img_smoothed * 255).astype(np.uint8)

# These thresholds (10, 30) are standard for your SimCATS data
# 'sigma=1.0' here is the Canny's internal smoothing, separate from ours
edges = canny(img_uint8, sigma=1.0, low_threshold=10, high_threshold=30)

# ── 4. Visualize the Preprocessing Pipeline ─────────────────
# Increased figure height slightly to make room for titles
fig, axs = plt.subplots(1, 3, figsize=(18, 7))

axs[0].imshow(img_gray, cmap="gray")
# Added fontsize and padding (pad=20)
axs[0].set_title("1. Grayscale Baseline", fontsize=16, pad=20)
axs[0].axis("off")

axs[1].imshow(img_smoothed, cmap="inferno")
axs[1].set_title("2. Smoothed (Sigma=2.0)", fontsize=16, pad=20)
axs[1].axis("off")

axs[2].imshow(edges, cmap="gray")
axs[2].set_title("3. Canny Edge Map (2D Input)", fontsize=16, pad=20)
axs[2].axis("off")

# tight_layout with a top rect adjustment to ensure titles aren't cut off
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

out_img = os.path.join(out_folder, "2D_preprocessing_gallery.png")
plt.savefig(out_img, dpi=200, bbox_inches="tight")
plt.close()

print(f"2D Preprocessing complete.")
print(f"Input : {input_path}")
print(f"Result: {out_img}")