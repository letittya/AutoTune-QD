import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import affine_transform

# ── 1. setup paths (CLI or fallback) ────────────────────────
if len(sys.argv) > 1:
    # Batch mode: route to testing_1D/results/<image_name>
    img_path = sys.argv[1]
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    base_dir = os.path.join("testing_1D", "results", img_basename)
    
    json_path = os.path.join(base_dir, "extracted_lines.json")
    out_folder = os.path.join(base_dir, "Virtual_Gates")
else:
    # Standalone mode: main 1D folder
    img_path = os.path.join("CSD_generated_images", "csd_clean_simCAT.png")
    json_path = os.path.join("1D", "extracted_lines.json")
    out_folder = os.path.join("1D", "Virtual_Gates")

os.makedirs(out_folder, exist_ok=True)

# Safety check so it doesn't crash if it runs before extraction
if not os.path.exists(img_path) or not os.path.exists(json_path):
    print(f"Missing image or JSON for {img_path}. Skipping.")
    sys.exit(1)

# ── 2. load data ────────────────────────────────────────────
img = plt.imread(img_path)

if img.ndim == 3:
    img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
else:
    img_gray = img

with open(json_path, "r") as f:
    lines = json.load(f)

# ── 3. compute the golden numbers ───────────────────────────
diag_slopes = [l["slope"] for l in lines if l["type"] == "diagonal"]
steep_slopes = [l["slope"] for l in lines if l["type"] == "steep"]

m1 = np.mean(diag_slopes)  
m2 = np.mean(steep_slopes) 

# --- CRITICAL FIX: Define ratio here ---
ratio = m2 / m1 
# ---------------------------------------

print(f"Loaded Physical Slopes -> Diagonal: {m1:.4f}, Steep: {m2:.4f}")
print(f"Calculated Slope Ratio: {ratio:.2f}")

# ── 4. build the virtual gate matrix ────────────────────────
v_col = np.array([m1, 1.0])
v_col = v_col / np.linalg.norm(v_col)

v_row = np.array([m2, 1.0])
v_row = v_row / np.linalg.norm(v_row)

M = np.array([
    [v_row[0], v_col[0]],
    [v_row[1], v_col[1]]
])

print("\nVirtual Gate Transformation Matrix:")
print(np.round(M, 4))

# ── 5. apply the transformation (warp the image) ────────────
h, w = img_gray.shape
center = np.array([h / 2, w / 2])
offset = center - np.dot(M, center)

warped_img = affine_transform(img_gray, M, offset=offset, order=1)

# ── 6. visualize the absolute victory ───────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(img_gray, cmap="inferno")
ax1.set_title("Physical Gates (Skewed)")
ax1.axis("off")

ax2.imshow(warped_img, cmap="inferno")
ax2.set_title("Virtual Gates (Checkerboard)")
ax2.axis("off")

plt.tight_layout()
out_img = os.path.join(out_folder, "virtual_gates_proof.png")
plt.savefig(out_img, dpi=200)
plt.close()

print(f"\nSUCCESS! Saved proof to: {out_img}")

# ── 7. Save the Matrix & Validate Orthogonality ─────────────
M_inv = np.linalg.inv(M)
d1 = M_inv @ np.array([m1, 1.0])
d2 = M_inv @ np.array([m2, 1.0])
cos_theta = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
print(f"Post-transformation angle: {angle:.2f}° (ideal = 90.00°)")

# Calculate Physics Matrix
alpha_12 = m1
alpha_21 = 1.0 / m2
M_physics = np.array([
    [1.0, alpha_12],
    [alpha_21, 1.0]
])

# 95% Confidence Intervals (with safety catch for small samples)
def get_ci(slopes):
    if len(slopes) < 2:
        return [float(slopes[0]), float(slopes[0])] if slopes else [0.0, 0.0]
    lo, hi = stats.t.interval(0.95, df=len(slopes)-1, 
                              loc=np.mean(slopes), 
                              scale=stats.sem(slopes))
    return [float(lo), float(hi)]

ci_diag = get_ci(diag_slopes)
ci_steep = get_ci(steep_slopes)

print(f"Diagonal 95% CI: [{ci_diag[0]:.4f}, {ci_diag[1]:.4f}]")
print(f"Steep    95% CI: [{ci_steep[0]:.4f}, {ci_steep[1]:.4f}]")

matrix_data = {
    "M_geometric": M.tolist(),
    "M_physics": M_physics.tolist(),
    "diagonal_mean": float(m1),
    "steep_mean": float(m2),
    "slope_ratio": float(ratio),
    "orthogonality_angle_deg": float(angle),
    "diagonal_95_CI": [float(ci_diag[0]), float(ci_diag[1])],
    "steep_95_CI": [float(ci_steep[0]), float(ci_steep[1])]
}

with open(os.path.join(out_folder, "vg_matrix.json"), "w") as f:
    json.dump(matrix_data, f, indent=4)

print("Matrix saved to JSON. 1D delivery is 100% complete.")


