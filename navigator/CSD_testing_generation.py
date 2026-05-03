"""
generates 10 completely new QArray CSD images for testing
Guaranteed unique seeds to prevent data leakage.
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

from qarray import DotArray, GateVoltageComposer, charge_state_to_scalar

# Target the new navigator folders
ROOT    = "navigator"
IMG_DIR = os.path.join(ROOT, "csd_images")
LBL_DIR = os.path.join(ROOT, "csd_labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# Constants
RESOLUTION   = 500
V_MIN, V_MAX = -4.0, 4.0
N_IMAGES     = 10       
WOBBLE_SIGMA = 15

# Core generator 
def generate_csd(cgd_off: float, wobble_amplitude: float, seed: int):
    Cdd = np.array([[0.0, 0.08], [0.08, 0.0]])
    Cgd = np.array([[1.0, cgd_off], [cgd_off, 1.0]])

    model    = DotArray(Cdd=Cdd, Cgd=Cgd,
                        algorithm="default", implementation="rust", T=0.05)
    composer = GateVoltageComposer(n_gate=model.n_gate)
    vg       = composer.do2d(
        x_gate=1, x_min=V_MIN, x_max=V_MAX, x_res=RESOLUTION,
        y_gate=2, y_min=V_MIN, y_max=V_MAX, y_res=RESOLUTION,
    )

    n_open = model.ground_state_open(vg)
    z_raw  = charge_state_to_scalar(n_open).astype(np.float64)

    # flip so origin is bottom-left in voltage space
    z_raw  = z_raw[::-1, ::-1].copy()
    n_open = n_open[::-1, ::-1, :].copy()

    # transition lines via gradient
    gz_y, gz_x = np.gradient(z_raw)
    grad_mag    = np.hypot(gz_x, gz_y)
    core        = gaussian_filter(grad_mag, sigma=1.0)
    glow        = gaussian_filter(grad_mag, sigma=3.5)
    lines       = 0.6 * core + 0.4 * glow

    rng = np.random.default_rng(seed=seed)

    charge_noise     = gaussian_filter(
        rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=5)   * 0.07
    background_drift = gaussian_filter(
        rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=200) * 0.015

    lines_norm = lines / (lines.max() + 1e-12)
    signal     = np.clip(lines_norm + 0.08 + background_drift + charge_noise, 0, None)
    signal    /= signal.max()

    # wobble
    disp_x = gaussian_filter(
        rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=WOBBLE_SIGMA) * wobble_amplitude
    disp_y = gaussian_filter(
        rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=WOBBLE_SIGMA) * wobble_amplitude

    row_coords, col_coords = np.mgrid[0:RESOLUTION, 0:RESOLUTION]

    # 1. warp the visual signal
    signal = map_coordinates(signal,
                             [row_coords + disp_y, col_coords + disp_x],
                             order=1, mode="nearest")

    # 2. warp n_open identically
    n_open_wobbled = np.zeros_like(n_open)
    n_open_wobbled[:, :, 0] = map_coordinates(
        n_open[:, :, 0].astype(float),
        [row_coords + disp_y, col_coords + disp_x],
        order=0, mode="nearest")
    n_open_wobbled[:, :, 1] = map_coordinates(
        n_open[:, :, 1].astype(float),
        [row_coords + disp_y, col_coords + disp_x],
        order=0, mode="nearest")

    signal_display = np.power(np.clip(signal, 0, 1), 0.55)

    # 3. flip vertically 
    signal_display = signal_display[::-1, :]
    n_open_wobbled = n_open_wobbled[::-1, :, :]

    return signal_display, n_open_wobbled

def save_clean_png(array: np.ndarray, path: str):
    plt.imsave(path, array, cmap="inferno")

def compute_pixel_labels(n_open_wobbled: np.ndarray) -> dict:
    n1 = n_open_wobbled[:, :, 0].astype(int)
    n2 = n_open_wobbled[:, :, 1].astype(int)

    labels = {}
    for raw_state in np.unique(n1 * 100 + n2):
        s1, s2 = divmod(int(raw_state), 100)
        mask   = (n1 == s1) & (n2 == s2)
        if int(mask.sum()) < 200:   
            continue
        row_idx, col_idx = np.where(mask)
        labels[f"{s1}_{s2}"] = {
            "col_px":      int(col_idx.mean()),
            "row_px":      int(row_idx.mean()),
            "pixel_count": int(mask.sum()),
        }
    return labels

# Brand new seeds to guarantee zero overlap with training data 
rng_main = np.random.default_rng(seed=8888) 

alphas  = rng_main.uniform(0.20, 0.40, size=N_IMAGES)
wobbles = rng_main.uniform(10.0, 45.0, size=N_IMAGES)

print(f"Generating {N_IMAGES} unseen testing images...\n{'─'*60}")

missing_11 = []

for i in range(N_IMAGES):
    alpha      = float(alphas[i])
    wobble     = float(wobbles[i])
    seed       = 9000 + i  # new seed offset
    
    img_name   = f"navtest_{i+1:03d}_alpha{alpha:.3f}_wobble{wobble:.1f}"
    img_path   = os.path.join(IMG_DIR, f"{img_name}.png")
    label_path = os.path.join(LBL_DIR, f"{img_name}.json")

    t0 = time.time()
    signal_display, n_open_wobbled = generate_csd(
        cgd_off=alpha, wobble_amplitude=wobble, seed=seed)

    save_clean_png(signal_display, img_path)

    pixel_labels = compute_pixel_labels(n_open_wobbled)
    target_11    = pixel_labels.get("1_1", None)

    label_data = {
        "image_name":   img_name + ".png",
        "alpha":        round(alpha,  4),
        "wobble_px":    round(wobble, 2),
        "seed":         seed,
        "image_size":   RESOLUTION,
        "target_state": "1_1",
        "target_pixel": target_11,
        "all_states":   pixel_labels,
    }

    with open(label_path, "w") as f:
        json.dump(label_data, f, indent=4)

    elapsed = time.time() - t0
    status  = "✓ (1,1) found" if target_11 else "✗ (1,1) MISSING"
    print(f"[{i+1:03d}/{N_IMAGES}] alpha={alpha:.3f}  wobble={wobble:.1f}px  "
          f"{status}  ({elapsed:.1f}s)")

    if not target_11:
        missing_11.append(img_name)

print(f"\n{'─'*60}")
print(f"Done.")
print(f"  Images  → {IMG_DIR}/")
print(f"  Labels  → {LBL_DIR}/")