"""
reads every image + JSON pair and extracts:
  - 1 positive patch  : centered on (1,1) "cell interior"
  - 1 hard negative   : centered on (0,0) "pure void"
  - 2 random negatives: random coords far from (1,1) 

so we will have about:
  100 positive patches   shape (N, 85, 85)
  300 negative patches   shape (3N, 85, 85)

Patches are grayscale.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# config 
ROOT       = "classifier_training"
IMG_DIR    = os.path.join(ROOT, "images")
LBL_DIR    = os.path.join(ROOT, "labels")
PATCH_DIR  = os.path.join(ROOT, "patches")
os.makedirs(PATCH_DIR, exist_ok=True)

PATCH_SIZE      = 85       # pixels — large enough to see a full diamond cell
HALF            = PATCH_SIZE // 2
MIN_DIST        = 100      # random negatives must be this far from (1,1)
MIN_BRIGHTNESS  = 0.08     # random negatives must have mean pixel above this
                           # to guarantee they hit a visible line, not pure void
MAX_RANDOM_TRIES = 200     # max attempts to find a bright random patch


#  helpers 
def load_image_gray(path: str) -> np.ndarray:
    """Load PNG and convert to grayscale float32 in [0,1]."""
    img = plt.imread(path)          # shape (H, W, 4) RGBA from plt.imsave
    if img.ndim == 3:
        # collapse RGBA → luminance using standard weights
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        gray = img.astype(np.float32)
    return gray.astype(np.float32)


def extract_patch(img: np.ndarray, col: int, row: int) -> np.ndarray | None:
    """
    Extract an 85x85 patch centered on (col, row).
    Returns None if the patch would go out of bounds.
    """
    r1 = row - HALF
    r2 = row + HALF + 1
    c1 = col - HALF
    c2 = col + HALF + 1

    # reject patches that touch the image border
    if r1 < 0 or r2 > img.shape[0] or c1 < 0 or c2 > img.shape[1]:
        return None

    return img[r1:r2, c1:c2]


def distance(col1, row1, col2, row2) -> float:
    return np.sqrt((col1 - col2) ** 2 + (row1 - row2) ** 2)


def find_random_line_patch(img: np.ndarray,
                            target_col: int, target_row: int,
                            rng: np.random.Generator) -> np.ndarray | None:
    """
    Sample random coordinates until we find a patch that:
      1. is far enough from (1,1)
      2. has mean brightness above MIN_BRIGHTNESS (hits a line)
      3. fits entirely within the image
    Returns the patch or None if no valid patch found after MAX_RANDOM_TRIES.
    """
    H, W = img.shape
    margin = HALF + 1   # keep patch fully inside image

    for _ in range(MAX_RANDOM_TRIES):
        col = int(rng.integers(margin, W - margin))
        row = int(rng.integers(margin, H - margin))

        # must be far from the target cell
        if distance(col, row, target_col, target_row) < MIN_DIST:
            continue

        patch = extract_patch(img, col, row)
        if patch is None:
            continue

        # must hit a visible line, not pure background
        if patch.mean() >= MIN_BRIGHTNESS:
            return patch

    return None   # give up  will be skipped with a warning


#  main loop 
label_files = sorted([f for f in os.listdir(LBL_DIR) if f.endswith(".json")])

all_positive = []
all_negative = []
skipped      = []

print(f"Extracting patches from {len(label_files)} images...\n{'─'*60}")

for lbl_file in label_files:
    lbl_path = os.path.join(LBL_DIR, lbl_file)
    with open(lbl_path, "r") as f:
        data = json.load(f)

    img_path = os.path.join(IMG_DIR, data["image_name"])
    if not os.path.exists(img_path):
        print(f"  SKIP (image missing): {data['image_name']}")
        skipped.append(lbl_file)
        continue

    img  = load_image_gray(img_path)
    rng  = np.random.default_rng(seed=data["seed"])
    states = data["all_states"]

    # positive: (1,1) cell interior 
    target = data.get("target_pixel")
    if target is None:
        print(f"  SKIP (no (1,1)): {data['image_name']}")
        skipped.append(lbl_file)
        continue

    t_col, t_row = target["col_px"], target["row_px"]
    pos_patch = extract_patch(img, t_col, t_row)
    if pos_patch is None:
        print(f"  SKIP (positive OOB): {data['image_name']}")
        skipped.append(lbl_file)
        continue

    # hard negative 1: (0,0) pure void 
    void = states.get("0_0")
    if void is None:
        print(f"  SKIP (no (0,0)): {data['image_name']}")
        skipped.append(lbl_file)
        continue

    neg_void = extract_patch(img, void["col_px"], void["row_px"])
    if neg_void is None:
        print(f"  SKIP ((0,0) OOB): {data['image_name']}")
        skipped.append(lbl_file)
        continue

    #  random negatives 1 & 2: must land on a visible line 
    neg_rand1 = find_random_line_patch(img, t_col, t_row, rng)
    neg_rand2 = find_random_line_patch(img, t_col, t_row, rng)

    if neg_rand1 is None or neg_rand2 is None:
        print(f"  WARN (couldn't find 2 bright random patches): {data['image_name']}")
        # still use what we have just skip the missing one
        rand_negs = [n for n in [neg_rand1, neg_rand2] if n is not None]
    else:
        rand_negs = [neg_rand1, neg_rand2]

    #  collect 
    all_positive.append(pos_patch)
    all_negative.append(neg_void)
    all_negative.extend(rand_negs)

    print(f"  ✓ {data['image_name']}  "
          f"pos=1  neg={1 + len(rand_negs)}  "
          f"pos_mean={pos_patch.mean():.3f}  void_mean={neg_void.mean():.3f}")

#  stack and save 
patches_pos = np.stack(all_positive, axis=0)   # (N, 85, 85)
patches_neg = np.stack(all_negative, axis=0)   # (3N, 85, 85)

labels_pos  = np.ones(len(patches_pos),  dtype=np.int32)
labels_neg  = np.zeros(len(patches_neg), dtype=np.int32)

all_patches = np.concatenate([patches_pos, patches_neg], axis=0)
all_labels  = np.concatenate([labels_pos,  labels_neg],  axis=0)

np.save(os.path.join(PATCH_DIR, "patches_positive.npy"), patches_pos)
np.save(os.path.join(PATCH_DIR, "patches_negative.npy"), patches_neg)
np.save(os.path.join(PATCH_DIR, "all_patches.npy"),      all_patches)
np.save(os.path.join(PATCH_DIR, "all_labels.npy"),       all_labels)

print(f"Done.")
print(f"  Positive patches : {len(patches_pos)}")
print(f"  Negative patches : {len(patches_neg)}")
print(f"  Total            : {len(all_patches)}")
print(f"  Skipped images   : {len(skipped)}")
print(f"  Saved to         : {PATCH_DIR}/")
print(f"\n  all_patches.npy shape : {all_patches.shape}")
print(f"  all_labels.npy  shape : {all_labels.shape}")