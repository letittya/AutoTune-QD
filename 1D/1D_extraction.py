import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

# ── setup paths ─────────────────────────────────────────────
img_folder = "CSD_generated_images"
img_name = "csd_clean.png"
input_path = os.path.join(img_folder, img_name)

out_folder = "1D"
os.makedirs(out_folder, exist_ok=True)

# ── check file ──────────────────────────────────────────────
if not os.path.exists(input_path):
    print(f"Cannot find {input_path}")
    exit()

# ── 1. load image ───────────────────────────────────────────
img = plt.imread(input_path)

# ── 2. grayscale conversion ─────────────────────────────────
if img.ndim == 3:
    img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
else:
    img_gray = img

# ── 3. smoothing ────────────────────────────────────────────
img_smoothed = gaussian_filter(img_gray, sigma=2.0)

# ── 4. slicing ──────────────────────────────────────────────
h, w = img_smoothed.shape
mid_row_idx = h // 2
mid_col_idx = w // 2

row_signal = img_smoothed[mid_row_idx, :]
col_signal = img_smoothed[:, mid_col_idx]

def detect_peaks(signal):
    prominence = 0.2 * (np.max(signal) - np.min(signal))
    return find_peaks(signal, prominence=prominence, distance=15)[0]

# ── 5. peak detection ───────────────────────────────────────
row_peaks = detect_peaks(row_signal)
col_peaks = detect_peaks(col_signal)

print(f"Detected row peaks: {len(row_peaks)}")
print(f"Detected col peaks: {len(col_peaks)}")

# ── preprocessing gallery ───────────────────────────────────
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(img)
axs[0].set_title("1. Raw input")
axs[0].axis("off")

axs[1].imshow(img_gray, cmap="gray")
axs[1].set_title("2. Grayscale")
axs[1].axis("off")

axs[2].imshow(img_smoothed, cmap="inferno")
axs[2].axhline(mid_row_idx, color="cyan", linestyle="--")
axs[2].axvline(mid_col_idx, color="lime", linestyle="--")
axs[2].set_title("3. Smoothed + slices")
axs[2].axis("off")

axs[3].plot(row_signal, label="row", color="blue")
axs[3].plot(col_signal, label="col", color="green", alpha=0.7)
axs[3].set_title("4. 1D signals")
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "preprocessing_gallery.png"), dpi=200)
plt.close()

print("Saved preprocessing gallery")

# ── 1D slice verification ───────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(row_signal, color="blue")
ax1.set_title(f"Horizontal slice (row {mid_row_idx})")
ax1.grid(True, alpha=0.3)

ax2.plot(col_signal, color="green")
ax2.set_title(f"Vertical slice (col {mid_col_idx})")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "1d_slice_verification.png"), dpi=200)
plt.close()

print("Saved 1D slice verification")

# ── peak detection visualization ────────────────────────────
fig3, axs = plt.subplots(2, 1, figsize=(8, 6))

axs[0].plot(row_signal, color="blue")
axs[0].scatter(row_peaks, row_signal[row_peaks], color="red", s=40)
axs[0].set_title("Row signal + detected peaks")

axs[1].plot(col_signal, color="green")
axs[1].scatter(col_peaks, col_signal[col_peaks], color="orange", s=40)
axs[1].set_title("Column signal + detected peaks")

for ax in axs:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "peak_detection.png"), dpi=200)
plt.close()

print("Saved peak detection visualization")

# ── overlay on image ────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(6, 6))

ax.imshow(img_smoothed, cmap="inferno")

# row peaks
ax.scatter(row_peaks,
           np.full_like(row_peaks, mid_row_idx),
           color="red", s=30, label="row peaks")

# column peaks
ax.scatter(np.full_like(col_peaks, mid_col_idx),
           col_peaks,
           color="cyan", s=30, label="column peaks")

# slice lines
ax.axhline(mid_row_idx, color="white", linestyle="--", alpha=0.5)
ax.axvline(mid_col_idx, color="white", linestyle="--", alpha=0.5)

ax.set_title("Detected peaks on CSD")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "detections_overlay.png"), dpi=200)
plt.close()

print("Saved overlay with detected peaks")






# ── MULTI-SLICE DETECTION (NEXT STEP) ───────────────────────

all_points = []

# choose slices (avoid edges)
margin = int(0.2 * h)
row_indices = np.arange(margin, h - margin, 5)

for r in row_indices:
    signal = img_smoothed[r, :]

    peaks = detect_peaks(signal)

    for p in peaks:
        all_points.append((p, r))  # (x, y)


# ── ADD COLUMN SLICING (FIX FOR MISSING LINES) ──────────────

col_indices = np.arange(margin, w - margin, 5)

for c in col_indices:
    signal = img_smoothed[:, c]

    peaks = detect_peaks(signal)

    for p in peaks:
        all_points.append((c, p))  # (x, y)


all_points = np.array(all_points)
all_points = np.unique(all_points, axis=0)

print(f"Total detected points from multi-slice: {len(all_points)}")


# ── visualize multi-slice points ────────────────────────────
fig6, ax = plt.subplots(figsize=(6, 6))

ax.imshow(img_smoothed, cmap="inferno")

if len(all_points) > 0:
    ax.scatter(all_points[:, 0], all_points[:, 1],
               s=10, color="cyan", label="multi-slice peaks")

ax.set_title("Multi-slice detected points")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "multi_slice_points.png"), dpi=200)
plt.close()

print("Saved multi-slice visualization")



# ── STEP 1: SINGLE RANSAC LINE ─────────────────────────────
from sklearn.linear_model import RANSACRegressor

X = all_points[:, 0].reshape(-1, 1)  # x (columns)
y = all_points[:, 1]                 # y (rows)

ransac = RANSACRegressor(
    residual_threshold=5.0,  # how "thick" the line can be
    max_trials=500,
    random_state=0
)

ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask

print(f"Inliers: {inlier_mask.sum()} / {len(all_points)}")

# slope
slope = ransac.estimator_.coef_[0]
intercept = ransac.estimator_.intercept_

print(f"Slope: {slope:.4f}")

# ── visualize single RANSAC fit ────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_smoothed, cmap="inferno")

# inliers (line)
ax.scatter(all_points[inlier_mask, 0],
           all_points[inlier_mask, 1],
           s=6, color="lime", label="RANSAC inliers")

# outliers (everything else)
ax.scatter(all_points[outlier_mask, 0],
           all_points[outlier_mask, 1],
           s=3, color="white", alpha=0.2, label="outliers")

# draw fitted line
x_line = np.array([0, w])
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color="red", linewidth=2, label="fitted line")

ax.set_title("Single RANSAC line")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "ransac_single.png"), dpi=200)
plt.close()

print("Saved single RANSAC result")


# ── STEP 2: ITERATIVE RANSAC ───────────────────────────────
from sklearn.linear_model import RANSACRegressor

remaining_points = all_points.copy()
line_clusters = []

MIN_INLIERS = 80   # 🔑 key parameter (prevents tiny fake lines)

while len(remaining_points) > MIN_INLIERS:

    X = remaining_points[:, 0].reshape(-1, 1)
    y = remaining_points[:, 1]

    ransac = RANSACRegressor(
        residual_threshold=5.0,
        max_trials=500,
        random_state=0
    )

    try:
        ransac.fit(X, y)
    except ValueError:
        break

    inlier_mask = ransac.inlier_mask_
    n_inliers = inlier_mask.sum()

    # stop if line is too small (important)
    if n_inliers < MIN_INLIERS:
        break

    # save this line
    inliers = remaining_points[inlier_mask]
    line_clusters.append(inliers)

    print(f"Found line {len(line_clusters)} with {n_inliers} points")

    # remove these points
    remaining_points = remaining_points[~inlier_mask]

print(f"\nTotal lines found: {len(line_clusters)}")


# ── visualize iterative lines ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_smoothed, cmap="inferno")

colors = plt.cm.tab10(np.linspace(0, 1, max(len(line_clusters), 1)))

for i, pts in enumerate(line_clusters):
    ax.scatter(pts[:, 0], pts[:, 1],
               s=6, color=colors[i],
               label=f"Line {i+1}")

# leftover noise
if len(remaining_points) > 0:
    ax.scatter(remaining_points[:, 0], remaining_points[:, 1],
               s=3, color="white", alpha=0.2, label="noise")

ax.set_title(f"Iterative RANSAC ({len(line_clusters)} lines)")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "ransac_iterative.png"), dpi=200)
plt.close()

print("Saved iterative RANSAC result")

# ── STEP 3: FILTER LINES BY SLOPE ──────────────────────────

filtered_lines = []
line_slopes = []

for pts in line_clusters:
    # compute slope using linear fit
    coef = np.polyfit(pts[:, 0], pts[:, 1], 1)
    slope = coef[0]

    line_slopes.append(slope)

    # keep only diagonal-ish lines (remove vertical ones)
    if abs(slope) < 2.0:   # 🔑 threshold
        filtered_lines.append(pts)

print(f"Filtered lines: {len(filtered_lines)} / {len(line_clusters)}")


# ── VISUALIZE FILTERED LINES ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_smoothed, cmap="inferno")

colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_lines)))

for i, pts in enumerate(filtered_lines):
    ax.scatter(pts[:, 0], pts[:, 1],
               s=6, color=colors[i],
               label=f"Line {i+1}")

ax.set_title(f"Filtered diagonal lines ({len(filtered_lines)})")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "filtered_lines.png"), dpi=200)
plt.close()

print("Saved filtered lines")


# ── STEP 4: EXTRACT FINAL SLOPES + METADATA ────────────────

final_data = []

for i, pts in enumerate(filtered_lines):
    # fit line again cleanly
    coef = np.polyfit(pts[:, 0], pts[:, 1], 1)
    slope = coef[0]
    intercept = coef[1]

    final_data.append({
        "line_id": i + 1,
        "slope": float(slope),
        "intercept": float(intercept),
        "num_points": len(pts)
    })

    print(f"Line {i+1}: slope = {slope:.4f}, points = {len(pts)}")


# ── SAVE RESULTS ───────────────────────────────────────────
import json

with open(os.path.join(out_folder, "extracted_lines.json"), "w") as f:
    json.dump(final_data, f, indent=4)

print("Saved line data → extracted_lines.json")




