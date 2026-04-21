"""
USING simcats to try to simulate csds 
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# simcats api public
from simcats import Simulation, default_configs


from simcats.distortions import OccupationTransitionBlurringGaussian

# config

cfg = default_configs["GaAs_v1"]

VOLT_LIMITS_G1 = cfg["volt_limits_g1"]   # np.array([-0.2,  -0.087])
VOLT_LIMITS_G2 = cfg["volt_limits_g2"]   # np.array([-0.2,  -0.047])

RESOLUTION = 500   # pixels per axis → (500, 500) image


sim = Simulation(
    volt_limits_g1=VOLT_LIMITS_G1,
    volt_limits_g2=VOLT_LIMITS_G2,
    ideal_csd_config=cfg["ideal_csd_config"],
    sensor=cfg["sensor"],
    occupation_distortions=[
        OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100),
    ],
)


#  measure


csd, occupations, lead_transitions, metadata = sim.measure(
    sweep_range_g1=VOLT_LIMITS_G1,
    sweep_range_g2=VOLT_LIMITS_G2,
    resolution=np.array([RESOLUTION, RESOLUTION]),
)


lt = (lead_transitions > 0).astype(np.float64)   # binary: line vs background

core = gaussian_filter(lt, sigma=1.0)
glow = gaussian_filter(lt, sigma=3.5)
lines = 0.6 * core + 0.4 * glow
lines_norm = lines / (lines.max() + 1e-12)


rng = np.random.default_rng(seed=42)
charge_noise = gaussian_filter(
    rng.normal(0, 1, (RESOLUTION, RESOLUTION)), sigma=5
) * 0.04

signal = lines_norm + 0.06 + charge_noise
signal = np.clip(signal, 0, None) / signal.max()
signal_display = np.power(signal, 0.55)   # gamma stretch

#plot
fig, ax = plt.subplots(figsize=(7, 6.5))

im = ax.imshow(
    signal_display,
    extent=[VOLT_LIMITS_G1[0], VOLT_LIMITS_G1[1],
            VOLT_LIMITS_G2[0], VOLT_LIMITS_G2[1]],
    origin="lower",
    aspect="auto",
    interpolation="bilinear",
    cmap="inferno",
    vmin=0.0,
    vmax=1.0,
)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Charge Transition Signal", fontsize=10)
cbar.set_ticks([0.0, 0.5, 1.0])
cbar.set_ticklabels(["low", "mid", "high"])

ax.set_title("Double Quantum Dot - Charge Stability Diagram (SimCATS GaAs_v1)",
             fontweight="bold", fontsize=11, pad=10)
ax.set_xlabel("V_G1  (V)", fontsize=11)
ax.set_ylabel("V_G2  (V)", fontsize=11)
ax.tick_params(labelsize=9)

fig.tight_layout()
fig.savefig("csd_simcats.png", dpi=200, bbox_inches="tight")
print("Saved: csd_simcats.png")
print(f"CSD shape         : {csd.shape}")
print(f"Lead transitions  : {np.unique(lead_transitions).tolist()} unique values")
print(f"Volt g1           : {VOLT_LIMITS_G1}")
print(f"Volt g2           : {VOLT_LIMITS_G2}")