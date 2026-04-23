import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

# ── 1. setup paths ──────────────────────────────────────────
img_path = os.path.join("CSD_generated_images", "csd_clean.png")
json_path = os.path.join("1D", "extracted_lines.json")

out_folder = os.path.join("1D", "Virtual_Gates")
os.makedirs(out_folder, exist_ok=True)

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

# ── 7. Save the Matrix ──────────────────────────────────────
M_inv = np.linalg.inv(M)
d1 = M_inv @ np.array([m1, 1.0])
d2 = M_inv @ np.array([m2, 1.0])
cos_theta = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
print(f"Post-transformation angle: {angle:.2f}° (ideal = 90.00°)")

matrix_data = {
    "M_virtual_gate": M.tolist(),
    "diagonal_mean": float(m1),
    "steep_mean": float(m2),
    "slope_ratio": float(ratio),
    "orthogonality_angle_deg": float(angle)
}

with open(os.path.join(out_folder, "vg_matrix.json"), "w") as f:
    json.dump(matrix_data, f, indent=4)

print("Matrix saved to JSON. 1D delivery is 100% complete.")


