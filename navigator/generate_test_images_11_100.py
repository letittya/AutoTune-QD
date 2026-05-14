"""
uses the same seed (8888) as CSD_testing_generation and then discards the first 10
so images 11-100 are guaranteed consistent with the original 10.

seed:
  SVM training data : seeds 1000-1099  
  Test images 1-10  : seeds 9000-9009 
  Test images 11-100: seeds 9010-9099 
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

ROOT    = "navigator"
IMG_DIR = os.path.join(ROOT, "csd_images")
LBL_DIR = os.path.join(ROOT, "csd_labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

RESOLUTION   = 500
V_MIN, V_MAX = -4.0, 4.0
WOBBLE_SIGMA = 15


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

    z_raw  = z_raw[::-1, ::-1].copy()
    n_open = n_open[::-1, ::-1, :].copy()

    gz_y, gz_x = np.gradient(z_raw)
    grad_mag    = np.hypot(gz_x, gz_y)
    core        = gaussian_filter(grad_mag, sigma=1.0)
    glow        = gaussian_filter(grad_mag, sigma=3.5)
    lines       = 0.6 * core + 0.4 * glow

    rng = np.random.default_rng(seed=seed)

    charge_noise     = gaussian_filter(
        rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=5) * 0.07
    background_drift = gaussian_filter(
        rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=200) * 0.015

    lines_norm = lines / (lines.max() + 1e-12)
    signal     = np.clip(lines_norm + 0.08 + background_drift + charge_noise, 0, None)
    signal    /= signal.max()

    disp_x = gaussian_filter(
        rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=WOBBLE_SIGMA) * wobble_amplitude
    disp_y = gaussian_filter(
        rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=WOBBLE_SIGMA) * wobble_amplitude

    row_coords, col_coords = np.mgrid[0:RESOLUTION, 0:RESOLUTION]

    signal = map_coordinates(signal,
                             [row_coords + disp_y, col_coords + disp_x],
                             order=1, mode="nearest")

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

    signal_display = signal_display[::-1, :]
    n_open_wobbled = n_open_wobbled[::-1, :, :]

    return signal_display, n_open_wobbled


def save_clean_png(array: np.ndarray, path: str):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(array, origin="upper", aspect="equal",
              interpolation="bilinear", cmap="inferno", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)


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


# full 100-sample sequence from seed 8888, skip the first 10
rng_main    = np.random.default_rng(seed=8888)
alphas_all  = rng_main.uniform(0.20, 0.40, size=100)
wobbles_all = rng_main.uniform(10.0, 45.0, size=100)

alphas  = alphas_all[10:]   # indices 10-99 → images 11-100
wobbles = wobbles_all[10:]
N_NEW   = len(alphas)       # 90

print(f"Generating test images 11-100 ({N_NEW} images)...\n{'─'*60}")

missing_11 = []

for i in range(N_NEW):
    img_idx    = 11 + i          # 11 to 100
    alpha      = float(alphas[i])
    wobble     = float(wobbles[i])
    seed       = 9010 + i        # 9010 to 9099

    img_name   = f"navtest_{img_idx:03d}_alpha{alpha:.3f}_wobble{wobble:.1f}"
    img_path   = os.path.join(IMG_DIR, f"{img_name}.png")
    label_path = os.path.join(LBL_DIR, f"{img_name}.json")

    if os.path.exists(img_path) and os.path.exists(label_path):
        print(f"[{img_idx:03d}/100] SKIP (already exists): {img_name}")
        continue

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
    status  = " (1,1) found" if target_11 else " (1,1) MISSING"
    print(f"[{img_idx:03d}/100] alpha={alpha:.3f}  wobble={wobble:.1f}px  "
          f"{status}  ({elapsed:.1f}s)")

    if not target_11:
        missing_11.append(img_name)

print(f"\n{'─'*60}")
print(f"Done. Images 11-100 saved to:")
print(f"  Images  → {IMG_DIR}/")
print(f"  Labels  → {LBL_DIR}/")
if missing_11:
    print(f"\n  WARNING: (1,1) not found in {len(missing_11)} images:")
    for m in missing_11:
        print(f"    {m}")
else:
    print(f"  All {N_NEW} new images have a valid (1,1) target.")
