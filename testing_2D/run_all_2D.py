"""

Runs 2D_extraction.py on every img.
Each image gets its own subfolder.
"""

import os
import glob
import subprocess
import sys

# 1. grab images from the testing_1D folder
IMG_DIR = os.path.join("testing_1D", "images")

# 2. point to script inside the 2D folder
SCRIPT = os.path.join("2D", "2D_extraction.py")

images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))

if not images:
    print(f"No images found in {IMG_DIR}. Check your folder structure.")
    sys.exit(1)

print(f"Found {len(images)} images. Running 2D pipeline on each...\n")

for i, img_path in enumerate(images, 1):
    name = os.path.basename(img_path)
    print(f"[{i:02d}/{len(images)}]  {name}")
    print(f"{'─'*50}")

    # run the 2D extraction script
    result = subprocess.run(
        [sys.executable, SCRIPT, img_path],
        capture_output=False,   # let stdout print live
    )

    if result.returncode != 0:
        print(f"  !! FAILED with return code {result.returncode}\n")
    else:
        print(f"  ✓ done\n")

print("All 2D extractions and Virtual Gate transformations complete.")
print("Results can be found in: testing_2D/results/<image_name>/")