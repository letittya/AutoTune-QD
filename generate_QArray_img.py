"""
generate 1d test images only.
runs two batteries (alpha sweep and wobble sweep) 
and saves the clean pngs to testing_1D/images.
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

# qarray stuff
from qarray import DotArray, GateVoltageComposer, charge_state_to_scalar

# setup folders
ROOT      = "testing_1D"
IMG_DIR   = os.path.join(ROOT, "images")
os.makedirs(IMG_DIR, exist_ok=True)

RESOLUTION = 500
V_MIN, V_MAX = -4.0, 4.0

# 1. image generator
def generate_qarray_clean(cgd_off: float,
                           wobble_amplitude: float = 35,
                           seed: int = 7) -> np.ndarray:
    """
    spits out a clean csd array without matplotlib borders.
    """
    Cdd = np.array([[0.0, 0.08], [0.08, 0.0]])
    Cgd = np.array([[1.0, cgd_off], [cgd_off, 1.0]])

    model = DotArray(Cdd=Cdd, Cgd=Cgd,
                     algorithm="default", implementation="rust", T=0.05)

    composer = GateVoltageComposer(n_gate=model.n_gate)
    vg = composer.do2d(
        x_gate=1, x_min=V_MIN, x_max=V_MAX, x_res=RESOLUTION,
        y_gate=2, y_min=V_MIN, y_max=V_MAX, y_res=RESOLUTION,
    )

    n_open = model.ground_state_open(vg)
    z_raw  = charge_state_to_scalar(n_open).astype(np.float64)
    z_raw  = z_raw[::-1, ::-1].copy()

    gz_y, gz_x = np.gradient(z_raw)
    grad_mag    = np.hypot(gz_x, gz_y)
    core        = gaussian_filter(grad_mag, sigma=1.0)
    glow        = gaussian_filter(grad_mag, sigma=3.5)
    lines       = 0.6 * core + 0.4 * glow

    rng = np.random.default_rng(seed=seed)

    charge_noise     = gaussian_filter(rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=5)  * 0.07
    background_drift = gaussian_filter(rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=200) * 0.015

    lines_norm = lines / (lines.max() + 1e-12)
    signal     = np.clip(lines_norm + 0.08 + background_drift + charge_noise, 0, None)
    signal    /= signal.max()

    # add wobble so it mimics lab data
    WOBBLE_SIGMA = 15
    disp_x = gaussian_filter(rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=WOBBLE_SIGMA) * wobble_amplitude
    disp_y = gaussian_filter(rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=WOBBLE_SIGMA) * wobble_amplitude

    row_coords, col_coords = np.mgrid[0:RESOLUTION, 0:RESOLUTION]
    signal = map_coordinates(signal,
                             [row_coords + disp_y, col_coords + disp_x],
                             order=1, mode="nearest")

    signal = np.power(np.clip(signal, 0, 1), 0.55)
    return signal


def save_clean_png(array: np.ndarray, path: str):
    """save pure pixels"""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.imshow(array, origin="lower", aspect="equal",
              interpolation="bilinear", cmap="inferno", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# 2. test configs
ALPHA_TESTS = [
    # (id, name, cgd_off, wobble_amp, ground_truth_alpha)
    (1,  "alpha_0.10", 0.10, 35, 0.10),
    (2,  "alpha_0.15", 0.15, 35, 0.15),
    (3,  "alpha_0.20", 0.20, 35, 0.20),
    (4,  "alpha_0.25", 0.25, 35, 0.25),   # baseline
    (5,  "alpha_0.30", 0.30, 35, 0.30),
    (6,  "alpha_0.40", 0.40, 35, 0.40),
]

WOBBLE_TESTS = [
    (7,  "wobble_0",   0.25, 0,  0.25),
    (8,  "wobble_10",  0.25, 10, 0.25),
    (9,  "wobble_20",  0.25, 20, 0.25),
    (10, "wobble_35",  0.25, 35, 0.25),   # baseline duplicate
    (11, "wobble_45",  0.25, 45, 0.25),
    (12, "wobble_60",  0.25, 60, 0.25),
]

ALL_TESTS = ALPHA_TESTS + WOBBLE_TESTS


# 3. run loop to just generate images
for (idx, desc, cgd_off, wobble_amp, gt_alpha) in ALL_TESTS:
    name     = f"{idx}_{desc}"
    img_path = os.path.join(IMG_DIR, f"{name}.png")

    print(f"\n{'-'*60}")
    print(f"[{idx:02d}/{len(ALL_TESTS)}]  generating {name}")
    print(f"  cgd_off={cgd_off}  wobble={wobble_amp}")

    # generate
    t0    = time.time()
    array = generate_qarray_clean(cgd_off=cgd_off, wobble_amplitude=wobble_amp)
    save_clean_png(array, img_path)
    gen_time = time.time() - t0
    print(f"  image saved  ({gen_time:.1f}s)  -> {img_path}")

print("\nall images generated successfully.")