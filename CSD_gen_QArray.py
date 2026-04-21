import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

from qarray import DotArray, GateVoltageComposer, charge_state_to_scalar

# make a folder for the outputs so it doesnt clutter the main directory
out_dir = "CSD_generated_images"
os.makedirs(out_dir, exist_ok=True)

# setting up the physics model and capacitance matrices
Cdd = np.array([[0.0, 0.08], [0.08, 0.0]])
Cgd = np.array([[1.0, 0.25], [0.25, 1.0]])

model = DotArray(
    Cdd=Cdd, Cgd=Cgd,
    algorithm="default", implementation="rust", T=0.05,
)

RESOLUTION = 500
V_MIN, V_MAX = -4.0, 4.0

# sweep the voltages
composer = GateVoltageComposer(n_gate=model.n_gate)
vg = composer.do2d(
    x_gate=1, x_min=V_MIN, x_max=V_MAX, x_res=RESOLUTION,
    y_gate=2, y_min=V_MIN, y_max=V_MAX, y_res=RESOLUTION,
)

# get the ground state
n_open = model.ground_state_open(vg)
z_raw  = charge_state_to_scalar(n_open).astype(np.float64)

# fix the polarity so the origin is exactly bottom left
z_raw  = z_raw[::-1, ::-1].copy()
n_open = n_open[::-1, ::-1, :].copy()

# extract the transition lines using gradients
gz_y, gz_x = np.gradient(z_raw)
grad_mag    = np.hypot(gz_x, gz_y)
core        = gaussian_filter(grad_mag, sigma=1.0)
glow        = gaussian_filter(grad_mag, sigma=3.5)
lines       = 0.6 * core + 0.4 * glow

# add some background noise and drift so it looks like real lab data
rng = np.random.default_rng(seed=7)

charge_noise = gaussian_filter(
    rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=5
) * 0.07

background_drift = gaussian_filter(
    rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=200 
) * 0.015 

lines_norm = lines / (lines.max() + 1e-12)
signal     = lines_norm + 0.08 + background_drift + charge_noise
signal     = np.clip(signal, 0, None)
signal     = signal / signal.max()

# warp the image a bit to simulate experimental wobble
WOBBLE_SIGMA     = 15     # px spatial scale
WOBBLE_AMPLITUDE = 35    # px shift magnitude

disp_x = gaussian_filter(
    rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=WOBBLE_SIGMA
) * WOBBLE_AMPLITUDE

disp_y = gaussian_filter(
    rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=WOBBLE_SIGMA
) * WOBBLE_AMPLITUDE

row_coords, col_coords = np.mgrid[0:RESOLUTION, 0:RESOLUTION]
signal_warped = map_coordinates(
    signal,
    [row_coords + disp_y, col_coords + disp_x],
    order=1,
    mode="nearest",
)

# tweak the contrast with a gamma stretch
signal_display = np.power(np.clip(signal_warped, 0, 1), 0.55)

# calc the centroids for the charge states in case we need the labels back later
n1 = n_open[:, :, 0].astype(int)
n2 = n_open[:, :, 1].astype(int)

vx_pixels = np.linspace(V_MIN, V_MAX, RESOLUTION)   
vy_pixels = np.linspace(V_MIN, V_MAX, RESOLUTION)   

label_positions = {}
for raw_state in np.unique(n1 * 100 + n2):
    s1, s2 = divmod(int(raw_state), 100)
    mask   = (n1 == s1) & (n2 == s2)
    if int(mask.sum()) < 200:
        continue
    row_idx, col_idx = np.where(mask)
    label_positions[(s1, s2)] = (
        float(vx_pixels[col_idx].mean()),
        float(vy_pixels[row_idx].mean()),
    )

# save the raw clean array without the matplotlib axes for the slicing test
clean_path = os.path.join(out_dir, "csd_clean.png")
plt.imsave(clean_path, signal_display, cmap="inferno", origin="lower")
print(f"saved clean array: {clean_path}")

# plot the final figure with all the matplotlib styling
fig, ax = plt.subplots(figsize=(7, 6.5))

im = ax.imshow(
    signal_display,
    extent=[V_MIN, V_MAX, V_MIN, V_MAX],
    origin="lower", aspect="equal",
    interpolation="bilinear",
    cmap="inferno", vmin=0.0, vmax=1.0,
)

# commented out the labels for now bc we dont need them
"""for (s1, s2), (lx, ly) in label_positions.items():
    ax.text(lx, ly, "({},{})".format(s1, s2),
            color="white", fontsize=7.5, fontweight="bold",
            ha="center", va="center", alpha=0.85)"""

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Charge Transition Signal", fontsize=10)
cbar.set_ticks([0.0, 0.5, 1.0])
cbar.set_ticklabels(["low", "mid", "high"])

ax.set_title("Double Quantum Dot - QArray Charge Stability Diagram",
             fontweight="bold", fontsize=12, pad=10)
ax.set_xlabel("V_P1  (mV)", fontsize=11)
ax.set_ylabel("V_P2  (mV)", fontsize=11)
ax.tick_params(labelsize=9)

fig.tight_layout()
plot_path = os.path.join(out_dir, "csd_QArray.png")
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"saved plot: {plot_path}")