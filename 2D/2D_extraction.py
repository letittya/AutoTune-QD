import os
import sys
import json
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from sklearn.linear_model import RANSACRegressor
from scipy.ndimage import affine_transform

# step 1. path where to save img
# same img as i used for 1d
if len(sys.argv) > 1:
    input_path = sys.argv[1]
    img_basename = os.path.splitext(os.path.basename(input_path))[0]
    out_folder = os.path.join("testing_2D", "results", img_basename)
else:
    input_path = os.path.join("CSD_generated_images", "csd_clean.png")
    out_folder = "2D"

os.makedirs(out_folder, exist_ok=True)

if not os.path.exists(input_path):
    print(f"Cannot find {input_path}")
    sys.exit(1)

# step 2: identical to 1D, preprocessing steps
img = plt.imread(input_path)

# grayscale
if img.ndim == 3:
    img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
else:
    img_gray = img

# smoothing (sigma=2.0)
img_smoothed = gaussian_filter(img_gray, sigma=2.0)

# step 3. Canny Edge Detection 
# convert to uint8 for standard Canny processing
img_uint8 = (img_smoothed * 255).astype(np.uint8)

# thresholds (10, 30) are standard for SimCATS data
# 'sigma=1.0' here is the Canny's internal smoothing, separate from ours
edges = canny(img_uint8, sigma=1.0, low_threshold=10, high_threshold=30)

# step 4. visualize
# increase height slightly to make room for titles
fig, axs = plt.subplots(1, 3, figsize=(18, 7))

axs[0].imshow(img_gray, cmap="gray")
axs[0].set_title("1. Grayscale Baseline", fontsize=16, pad=20)
axs[0].axis("off")

axs[1].imshow(img_smoothed, cmap="inferno")
axs[1].set_title("2. Smoothed (Sigma=2.0)", fontsize=16, pad=20)
axs[1].axis("off")

axs[2].imshow(edges, cmap="gray")
axs[2].set_title("3. Canny Edge Map (2D Input)", fontsize=16, pad=20)
axs[2].axis("off")

# tight_layout with a top rect adjustment to ensure titles aren't cut off
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

out_img = os.path.join(out_folder, "2D_preprocessing_gallery.png")
plt.savefig(out_img, dpi=200, bbox_inches="tight")
plt.close()

print(f"2D Preprocessing complete.")
print(f"Input : {input_path}")
print(f"Result: {out_img}")






# step 5: probabilistic hough transform

# lock the randomness globally so the hough transform is 100% reproducible

# np.random.seed(0)  
lines = probabilistic_hough_line(
    edges,
    threshold=75,       # minimum votes to count as a line
    line_length=100,     # min length of a segment in pixels
    line_gap=25,        # max gap to bridge broken pixels
    rng=0 #for repoductibility 
)

print(f"found {len(lines)} raw line segments using hough transform")

# calculate slopes and avoid vertical div by zero
slopes = []
valid_lines = []

for line in lines:
    p0, p1 = line
    x0, y0 = p0
    x1, y1 = p1

    # check if line is perfectly vertical
    if abs(x1 - x0) > 1e-5:
        slope = (y1 - y0) / (x1 - x0)
        slopes.append(slope)
        valid_lines.append(line)
    else:
        continue # drop vertical lines to prevent np.inf from crashing downstream MAD filters

# visualize the hough line segments
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_smoothed, cmap="inferno")

# lock axes to the image size
h, w = img_smoothed.shape
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

# plot each detected segment
for line in valid_lines:
    p0, p1 = line
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color="lime", linewidth=2)

ax.set_title(f"hough transform ({len(valid_lines)} segments)", fontsize=14)
ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "hough_lines.png"), dpi=200, bbox_inches="tight")
plt.close()

print("saved hough lines visualization")





# step 6: classify into two families
diagonal_segments = []
steep_segments = []

for line, slope in zip(valid_lines, slopes):
    p0, p1 = line
    x0, y0 = p0
    
    # calculate intercepts for infinite line reconstruction later
    if np.isinf(slope):
        intercept = x0
    else:
        intercept = y0 - slope * x0
        
    segment_data = {
        "pts": [p0, p1],
        "slope": float(slope),
        "intercept": float(intercept)
    }
    
    # same classification rule as 1d
    if abs(slope) < 0.8:
        diagonal_segments.append(segment_data)
    else:
        steep_segments.append(segment_data)

print(f"raw diagonal segments: {len(diagonal_segments)}")
print(f"raw steep segments: {len(steep_segments)}")

# mad outlier rejection
diag_slopes_raw = [s["slope"] for s in diagonal_segments]
steep_slopes_raw = [s["slope"] for s in steep_segments]

diag_med = np.median(diag_slopes_raw)
diag_mad = np.median([abs(s - diag_med) for s in diag_slopes_raw])
diag_mad = max(diag_mad, 1e-5)

steep_med = np.median(steep_slopes_raw)
steep_mad = np.median([abs(s - steep_med) for s in steep_slopes_raw])
steep_mad = max(steep_mad, 1e-5)

mad_threshold = 3.5

clean_diagonal = [s for s in diagonal_segments if abs(s["slope"] - diag_med) < mad_threshold * diag_mad]
clean_steep = [s for s in steep_segments if abs(s["slope"] - steep_med) < mad_threshold * steep_mad]

print(f"clean diagonal segments: {len(clean_diagonal)}")
print(f"clean steep segments: {len(clean_steep)}")

# step 7: visualize the clean slope families

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_smoothed, cmap="inferno")

h, w = img_smoothed.shape
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

# plot diagonal segments in cyan
for seg in clean_diagonal:
    p0, p1 = seg["pts"]
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color="cyan", linewidth=2)

# plot steep segments in yellow
for seg in clean_steep:
    p0, p1 = seg["pts"]
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color="yellow", linewidth=2)

legend_elements = [
    Line2D([0], [0], color="cyan", linewidth=2, label="diagonal family"),
    Line2D([0], [0], color="yellow", linewidth=2, label="steep family")
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

ax.set_title(f"clean hough families ({len(clean_diagonal) + len(clean_steep)} segments)", fontsize=14)
ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "two_slope_families.png"), dpi=200, bbox_inches="tight")
plt.close()

print("saved clean hough families plot")





# step 8: group segments into physical lines and export matching json

def group_and_fit(segments, family_type, is_steep=False, center_val=None):
    if center_val is None:
        center_val = img_smoothed.shape[1] // 2 # find the image center instead of hardcoding it
    if not segments: return []
    
    for seg in segments:
        (x0, y0), (x1, y1) = seg["pts"]
        m = seg["slope"]
        if is_steep:
            seg["center_cross"] = x0 if np.isinf(m) else x0 + (1.0 / m) * (center_val - y0)
        else:
            seg["center_cross"] = y0 if m == 0 else y0 + m * (center_val - x0)

    segments = sorted(segments, key=lambda s: s["center_cross"])
    
    clusters = []
    current_cluster = [segments[0]]
    cluster_mean = segments[0]["center_cross"]
    
    # increased threshold to 70 and comparing to the group average
    for seg in segments[1:]:
        if abs(seg["center_cross"] - cluster_mean) < 70:
            current_cluster.append(seg)
            # update the running average
            cluster_mean = sum(s["center_cross"] for s in current_cluster) / len(current_cluster)
        else:
            clusters.append(current_cluster)
            current_cluster = [seg]
            cluster_mean = seg["center_cross"]
    clusters.append(current_cluster)
    
    physical_lines = []
    
    for cluster in clusters:
        xs, ys = [], []
        total_length = 0
        for seg in cluster:
            (x0, y0), (x1, y1) = seg["pts"]
            xs.extend([x0, x1])
            ys.extend([y0, y1])
            total_length += np.hypot(x1 - x0, y1 - y0)
            
        # changed to 500
        if total_length < 500:
            continue
            
        if is_steep:
            X = np.array(ys).reshape(-1, 1)
            y = np.array(xs)
        else:
            X = np.array(xs).reshape(-1, 1)
            y = np.array(ys)
            
        ransac = RANSACRegressor(residual_threshold=5.0, random_state=0)
        try:
            ransac.fit(X, y)
            raw_slope = float(ransac.estimator_.coef_[0])
            raw_intercept = float(ransac.estimator_.intercept_)
            
            if is_steep:
                true_slope = 1.0 / raw_slope
                true_intercept = -raw_intercept / raw_slope
            else:
                true_slope = raw_slope
                true_intercept = raw_intercept
                
            physical_lines.append({
                "type": family_type,
                "slope": true_slope,
                "intercept": true_intercept,
                "total_pixel_length": int(total_length), # accurate name internally
                # 'num_points' here just represents total pixel length to match 1D JSON
                "num_points": int(total_length) # keep for JSON compatibility with 1D
            })
        except ValueError:
            continue
            
    return physical_lines

# group the fragmented pieces back into the actual physical lines
final_diag_lines = group_and_fit(clean_diagonal, "diagonal", is_steep=False)
final_steep_lines = group_and_fit(clean_steep, "steep", is_steep=True)

# MAD FILTER 
def apply_mad_filter(physical_lines, threshold=3.5):
    if len(physical_lines) < 3:
        return physical_lines
    slopes = [l["slope"] for l in physical_lines]
    med_slope = np.median(slopes)
    mad = np.median([abs(s - med_slope) for s in slopes])
    mad = max(mad, 1e-5) # prevent divide by zero
    
    # keep only lines whose slope is within the MAD threshold
    return [l for l in physical_lines if abs(l["slope"] - med_slope) < threshold * mad]

# filter out the junk crossing lines
final_diag_lines = apply_mad_filter(final_diag_lines)
final_steep_lines = apply_mad_filter(final_steep_lines)

final_lines = final_diag_lines + final_steep_lines






# add line_id and force the exact dictionary key order as 1d
ordered_final_lines = []
for i, line in enumerate(final_lines):
    ordered_final_lines.append({
        "line_id": i + 1,
        "type": line["type"],
        "slope": line["slope"],
        "intercept": line["intercept"],
        "num_points": line["num_points"]
    })

final_lines = ordered_final_lines

# calculate average slopes from the reconstructed physical lines
diag_slopes = [l["slope"] for l in final_diag_lines]
steep_slopes = [l["slope"] for l in final_steep_lines]

m1 = np.mean(diag_slopes)
m2 = np.mean(steep_slopes)
ratio = m2 / m1

print(f"\nfinal 2d results:")
print(f"diagonal slope: {m1:.4f} (from {len(diag_slopes)} physical lines)")
print(f"steep slope: {m2:.4f} (from {len(steep_slopes)} physical lines)")
print(f"slope ratio (alpha product proxy): {ratio:.2f}")

# virtual gate matrix (geometric)
v_col = np.array([m1, 1.0])
v_col = v_col / np.linalg.norm(v_col)

v_row = np.array([m2, 1.0])
v_row = v_row / np.linalg.norm(v_row)

M = np.array([
    [v_row[0], v_col[0]],
    [v_row[1], v_col[1]]
])

# physics matrix
alpha_12 = m1
alpha_21 = 1.0 / m2
M_physics = np.array([
    [1.0, alpha_12],
    [alpha_21, 1.0]
])

# calculate orthogonality angle
M_inv = np.linalg.inv(M)
d1 = M_inv @ np.array([m1, 1.0])
d2 = M_inv @ np.array([m2, 1.0])
cos_theta = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

# proper confidence intervals using the physical lines
def get_ci(slopes_list):
    if len(slopes_list) < 2:
        return [float(slopes_list[0]), float(slopes_list[0])] if slopes_list else [0.0, 0.0]
    lo, hi = stats.t.interval(
        0.95, df=len(slopes_list)-1, 
        loc=np.mean(slopes_list), 
        scale=stats.sem(slopes_list)
    )
    return [float(lo), float(hi)]

ci_diag = get_ci(diag_slopes)
ci_steep = get_ci(steep_slopes)

matrix_data = {
    "M_geometric": M.tolist(),
    "M_physics": M_physics.tolist(),
    "diagonal_mean": float(m1),
    "steep_mean": float(m2),
    "slope_ratio": float(ratio),
    "orthogonality_angle_deg": float(angle),
    "diagonal_95_CI": ci_diag,
    "steep_95_CI": ci_steep
}

# save extracted lines in the exact 1d format
with open(os.path.join(out_folder, "extracted_lines.json"), "w") as f:
    json.dump(final_lines, f, indent=4)

vg_folder = os.path.join(out_folder, "Virtual_Gates")
os.makedirs(vg_folder, exist_ok=True)

with open(os.path.join(vg_folder, "vg_matrix.json"), "w") as f:
    json.dump(matrix_data, f, indent=4)



# vizualize lines
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_smoothed, cmap="inferno")

h_img, w_img = img_smoothed.shape
ax.set_xlim(0, w_img)
ax.set_ylim(h_img, 0)

# draw diagonal lines (cyan)
for line in final_diag_lines:
    x_vals = np.array([0, w_img])
    y_vals = line["slope"] * x_vals + line["intercept"]
    ax.plot(x_vals, y_vals, color="cyan", linewidth=2)
    
# draw steep lines (yellow)
for line in final_steep_lines:
    # for steep lines, calculate x from y so it draws perfectly top-to-bottom
    y_vals = np.array([0, h_img])
    x_vals = (y_vals - line["intercept"]) / line["slope"]
    ax.plot(x_vals, y_vals, color="yellow", linewidth=2)

legend_elements = [
    Line2D([0], [0], color="cyan", linewidth=2, label="diagonal physical lines"),
    Line2D([0], [0], color="yellow", linewidth=2, label="steep physical lines")
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
ax.set_title(f"reconstructed physical lines ({len(final_diag_lines)} diag, {len(final_steep_lines)} steep)", fontsize=14)
ax.axis("off")

plt.tight_layout()
out_img_lines = os.path.join(out_folder, "reconstructed_physical_lines.png")
plt.savefig(out_img_lines, dpi=200, bbox_inches="tight")
plt.close()

print(f"saved reconstructed physical lines plot to: {out_img_lines}")




# generate and save the virtual gates visual proof
h_img, w_img = img_gray.shape
center = np.array([h_img / 2, w_img / 2])
offset = center - np.dot(M, center)

warped_img = affine_transform(img_gray, M, offset=offset, order=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(img_gray, cmap="inferno")
ax1.set_title("Physical Gates (Skewed)", fontsize=14)
ax1.axis("off")

ax2.imshow(warped_img, cmap="inferno")
ax2.set_title("Virtual Gates (Checkerboard)", fontsize=14)
ax2.axis("off")

plt.tight_layout()
out_img_proof = os.path.join(vg_folder, "virtual_gates_proof.png")
plt.savefig(out_img_proof, dpi=200, bbox_inches="tight")
plt.close()


# generate the extra color-mapped physical lines legend 
fig, ax = plt.subplots(figsize=(9, 6)) 
ax.imshow(img_smoothed, cmap="inferno")

h_img, w_img = img_smoothed.shape
ax.set_xlim(0, w_img)
ax.set_ylim(h_img, 0)

num_lines = len(final_lines)
colors = plt.cm.tab20(np.linspace(0, 1, num_lines))

legend_elements = []

for idx, line in enumerate(final_lines):
    color = colors[idx]
    line_id = line["line_id"]
    family = line["type"]
    
    if family == "diagonal":
        x_vals = np.array([0, w_img])
        y_vals = line["slope"] * x_vals + line["intercept"]
    else: 
        y_vals = np.array([0, h_img])
        x_vals = (y_vals - line["intercept"]) / line["slope"]
        
    ax.plot(x_vals, y_vals, color=color, linewidth=2)
    
    legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=f"line {line_id} ({family})"))

ax.legend(handles=legend_elements, fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))

ax.set_title(f"reconstructed physical lines map ({num_lines} total)", fontsize=14)
ax.axis("off")

plt.tight_layout()
out_img_map = os.path.join(out_folder, "reconstructed_physical_lines_map.png")
plt.savefig(out_img_map, dpi=200, bbox_inches="tight")
plt.close()

print(f"saved color-mapped physical lines plot to: {out_img_map}")

print(f"saved proof to: {out_img_proof}")
print("\n2d pipeline complete. saved everything to your folders.")