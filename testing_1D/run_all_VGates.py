"""
run_all_VGates.py
─────────────
Runs virtual_gates.py on every PNG in testing_1D/images/.
Each output is saved into its corresponding subfolder:
testing_1D/results/<image_name>/Virtual_Gates/

Usage:
    python run_all_VGates.py
"""

import os
import glob
import subprocess
import sys

IMG_DIR    = os.path.join("testing_1D", "images")
SCRIPT     = os.path.join("1D", "virtual_gates.py")

images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))

if not images:
    print(f"No images found in {IMG_DIR}. Run your generator first.")
    sys.exit(1)

print(f"Found {len(images)} images. Computing Virtual Gates for each...\n")

for i, img_path in enumerate(images, 1):
    name = os.path.basename(img_path)
    print(f"[{i:02d}/{len(images)}]  {name}")
    print(f"{'─'*50}")

    result = subprocess.run(
        [sys.executable, SCRIPT, img_path],
        capture_output=False,   # let stdout print live
    )

    if result.returncode != 0:
        print(f"  !! FAILED with return code {result.returncode}\n")
    else:
        print(f"  ✓ done\n")

print("All virtual gate extractions complete.")
print("Results in: testing_1D/results/<image_name>/Virtual_Gates/")