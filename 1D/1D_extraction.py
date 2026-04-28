import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

# ── image path: CLI arg or fallback to original hardcoded ───
if len(sys.argv) > 1:
    # Batch mode: route outputs to testing_1D/results/<image_name>
    input_path = sys.argv[1]
    img_basename = os.path.splitext(os.path.basename(input_path))[0]
    out_folder = os.path.join("testing_1D", "results", img_basename)
else:
    # Standalone mode: just drop everything in the main 1D folder
    input_path = os.path.join("CSD_generated_images", "csd_clean_simCAT.png")
    out_folder = "1D"

os.makedirs(out_folder, exist_ok=True)

# ── check file ──────────────────────────────────────────────
if not os.path.exists(input_path):
    print(f"Cannot find {input_path}")
    sys.exit(1)

print(f"Input : {input_path}")
print(f"Output: {out_folder}/")

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


# ── STEP 2: ITERATIVE RANSAC (TWO-PASS) ────────────────────────
from sklearn.linear_model import RANSACRegressor

remaining_points = all_points.copy()
line_clusters = []

MIN_INLIERS = 60  

# 1. Define the rule: Pass 1 is ONLY allowed to find flat lines
def valid_flat_line(estimator, X, y):
    return abs(estimator.coef_[0]) < 0.8 #WAS 1.5

# --- PASS 1: Find Diagonal Lines (Normal X, Y) ---
print("--- PASS 1: Diagonal Lines ---")
while len(remaining_points) > MIN_INLIERS:
    X = remaining_points[:, 0].reshape(-1, 1)
    y = remaining_points[:, 1]

    # Add the rule directly to RANSAC
    ransac = RANSACRegressor(
        residual_threshold=8.0, 
        max_trials=1000, 
        random_state=0,
        is_model_valid=valid_flat_line # 🔥 The magic fix
    )
    
    try:
        ransac.fit(X, y)
    except ValueError:
        break # Breaks only if it completely runs out of valid math

    inlier_mask = ransac.inlier_mask_
    
    if inlier_mask.sum() < MIN_INLIERS:
        break # Breaks if the line is too short

    # Save the line
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    inliers = remaining_points[inlier_mask]
    
    line_clusters.append({
        "points": inliers, 
        "slope": slope, 
        "intercept": intercept, 
        "type": "diagonal"
    })
    print(f"Found diagonal line with {inlier_mask.sum()} points (slope: {slope:.2f})")
    
    # Delete points and repeat
    remaining_points = remaining_points[~inlier_mask]

# --- PASS 2: Find Steep Lines (Swapped Y, X) ---
print("\n--- PASS 2: Steep Lines (Swapped Axes) ---")
while len(remaining_points) > MIN_INLIERS:
    # SWAP X AND Y HERE!
    X_steep = remaining_points[:, 1].reshape(-1, 1) # Y becomes input
    y_steep = remaining_points[:, 0]                # X becomes target

    ransac = RANSACRegressor(residual_threshold=5.0, max_trials=500, random_state=0)
    try:
        ransac.fit(X_steep, y_steep)
    except ValueError:
        break

    inlier_mask = ransac.inlier_mask_
    
    if inlier_mask.sum() < MIN_INLIERS:
        break

    inliers = remaining_points[inlier_mask]
    
    # Calculate the TRUE slope and intercept back in the normal coordinate system
    swapped_slope = ransac.estimator_.coef_[0]
    swapped_intercept = ransac.estimator_.intercept_
    
    true_slope = 1.0 / swapped_slope
    true_intercept = -swapped_intercept / swapped_slope

    line_clusters.append({"points": inliers, "slope": true_slope, "intercept": true_intercept, "type": "steep"})
    print(f"Found steep line with {inlier_mask.sum()} points (slope: {true_slope:.2f})")
    
    remaining_points = remaining_points[~inlier_mask]

print(f"\nTotal lines found: {len(line_clusters)}")

# ── visualize iterative lines ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_smoothed, cmap="inferno")
h, w = img_smoothed.shape
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

colors = plt.cm.tab10(np.linspace(0, 1, max(len(line_clusters), 1)))

for i, line in enumerate(line_clusters):
    pts = line["points"]
    slope = line["slope"]
    intercept = line["intercept"]

    # original points
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color=colors[i])

    # 🔥 FULL THICK LINE
    x_line = np.array([0, w])
    y_line = slope * x_line + intercept

    # No valid mask needed, just plot it! The locked axes will crop it visually.
    ax.plot(x_line, y_line, color=colors[i], linewidth=2)

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

# ── STEP 3: CLASSIFY LINES INTO TWO FAMILIES ───────────────

diagonal_lines = []
steep_lines = []

for line in line_clusters:
    pts = line["points"]
    slope = line["slope"]   #  use stored value

    if abs(slope) < 1.0:
        diagonal_lines.append(line)
    else:
        steep_lines.append(line)

print(f"Diagonal lines: {len(diagonal_lines)}")
print(f"Steep lines: {len(steep_lines)}")

# ── visualize both slope families ──────────────────────────
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_smoothed, cmap="inferno")

# Lock axes so it perfectly frames the image
h, w = img_smoothed.shape
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

# diagonal lines (cyan)
for line in diagonal_lines:
    pts = line["points"]
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color="cyan")
    
    # Draw the solid line
    x_line = np.array([0, w])
    y_line = line["slope"] * x_line + line["intercept"]
    ax.plot(x_line, y_line, color="cyan", linewidth=2, alpha=0.6)

# steep lines (yellow)
for line in steep_lines:
    pts = line["points"]
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color="yellow")
    
    # Draw the solid line
    x_line = np.array([0, w])
    y_line = line["slope"] * x_line + line["intercept"]
    ax.plot(x_line, y_line, color="yellow", linewidth=2, alpha=0.6)

# --- ADD THE CUSTOM PHYSICS LEGEND ---
legend_elements = [
    Line2D([0], [0], color="cyan", linewidth=2, label="diagonal family (α₁₂)"),
    Line2D([0], [0], color="yellow", linewidth=2, label="steep family (α₂₁)")
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

ax.set_title("Two slope families (Physical Crosstalk)")

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "two_slope_families.png"), dpi=200)
plt.close()

print("Saved two slope families visualization with custom legend")


# ── STEP 4: EXTRACT BOTH SLOPE SETS ────────────────────────

final_data = []

# diagonal lines
for i, line in enumerate(diagonal_lines):
    pts = line["points"]
    slope = line["slope"]
    intercept = line["intercept"]

    final_data.append({
        "line_id": i + 1,
        "type": "diagonal",
        "slope": float(slope),
        "intercept": float(intercept),
        "num_points": len(pts)
    })

# steep lines
offset = len(diagonal_lines)

for i, line in enumerate(steep_lines):
    pts = line["points"]
    slope = line["slope"]
    intercept = line["intercept"]

    final_data.append({
        "line_id": offset + i + 1,
        "type": "steep",
        "slope": float(slope),
        "intercept": float(intercept),
        "num_points": len(pts)
    })


# ── PRINT FINAL RAW LINES ───────────────────────────

print("\nFinal extracted lines:")
for d in final_data:
    print(f"{d['type']:8s} | slope = {d['slope']:.4f} | points = {d['num_points']}")


# ── OPTIONAL: FILTER OUT SMALL JUNK LINES ───────────
MIN_POINTS = 60

filtered_lines = [l for l in final_data if l["num_points"] >= MIN_POINTS]

print("\nFiltered lines:")
for l in filtered_lines:
    print(f"{l['type']:8s} | slope = {l['slope']:.4f} | points = {l['num_points']}")





# ── SAVE RESULTS & ROBUST OUTLIER REJECTION (MAD) ───
import json

diag_raw = [l["slope"] for l in filtered_lines if l["type"] == "diagonal"]
steep_raw = [l["slope"] for l in filtered_lines if l["type"] == "steep"]

# Calculate Median and MAD for Diagonals
diag_med = np.median(diag_raw)
diag_mad = np.median([abs(s - diag_med) for s in diag_raw])
diag_mad = max(diag_mad, 1e-5) # Keep the 1e-5 safety fallback just to prevent divide-by-zero errors

# Calculate Median and MAD for Steep lines
steep_med = np.median(steep_raw)
steep_mad = np.median([abs(s - steep_med) for s in steep_raw])
steep_mad = max(steep_mad, 1e-5) # Safety fallback

# --- CLAUDE'S FIX: 3.5x MAD THRESHOLD ---
MAD_THRESHOLD = 3.5

# Keep only the lines that pass the 3.5x MAD filter
truly_filtered_lines = [
    l for l in filtered_lines 
    if (l["type"] == "diagonal" and abs(l["slope"] - diag_med) < MAD_THRESHOLD * diag_mad) or 
       (l["type"] == "steep" and abs(l["slope"] - steep_med) < MAD_THRESHOLD * steep_mad)
]

with open(os.path.join(out_folder, "extracted_lines.json"), "w") as f:
    json.dump(truly_filtered_lines, f, indent=4)

print(f"Rejected {len(filtered_lines) - len(truly_filtered_lines)} outlier junction lines via MAD.")
print("Saved clean line data → extracted_lines.json")


# ── REGENERATE CLEAN ITERATIVE PLOT (CLAUDE'S OPTION 2) ───
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_smoothed, cmap="inferno")
h, w = img_smoothed.shape
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

# Re-use the color mapping from earlier
colors = plt.cm.tab10(np.linspace(0, 1, max(len(line_clusters), 1)))

for i, line in enumerate(line_clusters):
    # Claude's Fix: Skip any line not in the clean set (match by slope)
    if not any(abs(line["slope"] - clean_l["slope"]) < 0.01 for clean_l in truly_filtered_lines):
        continue
        
    pts = line["points"]
    slope = line["slope"]
    intercept = line["intercept"]

    # Plot original points
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color=colors[i])
    
    # Plot thick line
    x_line = np.array([0, w])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=colors[i], linewidth=2)

ax.set_title(f"Iterative RANSAC (Cleaned: {len(truly_filtered_lines)} lines)")

plt.tight_layout()
# Saving as a NEW file so you keep both the raw and clean versions for your thesis
plt.savefig(os.path.join(out_folder, "ransac_iterative_clean.png"), dpi=200)
plt.close()

print("Saved regenerated clean iterative RANSAC plot")

diag_slopes = [l["slope"] for l in truly_filtered_lines if l["type"] == "diagonal"]
steep_slopes = [l["slope"] for l in truly_filtered_lines if l["type"] == "steep"]

print("\nFINAL RESULTS:")

print(f"Diagonal slope (mean): {np.mean(diag_slopes):.4f}")
print(f"Diagonal std: {np.std(diag_slopes):.4f}")
print(f"Steep slope (mean): {np.mean(steep_slopes):.4f}")
print(f"Steep std: {np.std(steep_slopes):.4f}")

ratio = np.mean(steep_slopes) / np.mean(diag_slopes)

print(f"Slope ratio (s2/s1 = α21/α12 product proxy): {ratio:.2f}")
print(f"  → diagonal α ≈ {np.mean(diag_slopes):.4f}  (input Cgd crosstalk was 0.25)")
print(f"  → steep α    ≈ {1/np.mean(steep_slopes):.4f}  (expected ~0.25 from symmetry)")
print("---------------------------------")
print("STATUS: 1D Feature Extraction Complete.")



