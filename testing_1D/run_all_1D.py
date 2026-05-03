"""

Runs 1D_extraction.py on every img.
Each image gets its own subfolder with all the usual plots + json.
 
"""
 
import os
import glob
import subprocess
import sys
 
IMG_DIR    = os.path.join("testing_1D", "images")
SCRIPT     = os.path.join("testing_1D", "1D_extraction.py")
 
images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
 
if not images:
    print(f"No images found in {IMG_DIR}. Run generate_test_images.py first.")
    sys.exit(1)
 
print(f"Found {len(images)} images. Running extraction on each...\n")
 
for i, img_path in enumerate(images, 1):
    name = os.path.basename(img_path)
    print(f"[{i:02d}/{len(images)}]  {name}")
    print(f"{'─'*50}")
 
    result = subprocess.run(
        [sys.executable, SCRIPT, img_path],
        capture_output=False,   # let stdout print live so you can watch progress
    )
 
    if result.returncode != 0:
        print(f"  !! FAILED with return code {result.returncode}\n")
    else:
        print(f"  ✓ done\n")
 
print("All extractions complete.")
print("Results in: testing_1D/results/<image_name>/")