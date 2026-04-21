import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ── setup paths ─────────────────────────────────────────────
img_folder = "CSD_generated_images"
img_name = "csd_clean.png"
input_path = os.path.join(img_folder, img_name)

out_folder = "1D"
os.makedirs(out_folder, exist_ok=True)

# ── check file ──────────────────────────────────────────────
if not os.path.exists(input_path):
    print(f"cant find {input_path}")
else:
    # 1. load raw image
    img = plt.imread(input_path)

    # 2. grayscale conversion
    if img.ndim == 3:
        img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        img_gray = img

    # 3. smoothing
    img_smoothed = gaussian_filter(img_gray, sigma=2.0)

    # 4. slicing
    h, w = img_smoothed.shape
    mid_row_idx = h // 2
    mid_col_idx = w // 2

    row_signal = img_smoothed[mid_row_idx, :]
    col_signal = img_smoothed[:, mid_col_idx]

    # ── preprocessing gallery ───────────────────────────────
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(img)
    axs[0].set_title("1. raw input")
    axs[0].axis("off")

    axs[1].imshow(img_gray, cmap="gray")
    axs[1].set_title("2. grayscale")
    axs[1].axis("off")

    axs[2].imshow(img_smoothed, cmap="inferno")
    axs[2].axhline(mid_row_idx, color="cyan", linestyle="--")
    axs[2].axvline(mid_col_idx, color="lime", linestyle="--")
    axs[2].set_title("3. smoothed + slices")
    axs[2].axis("off")

    axs[3].plot(row_signal, label="row", color="blue")
    axs[3].plot(col_signal, label="col", color="green", alpha=0.7)
    axs[3].set_title("4. 1D signals")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "preprocessing_gallery.png"), dpi=200)
    plt.close()

    print("saved preprocessing gallery")

# ── clean 1D slice plot ─────────────────────────────────────
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

print("saved 1D slice verification")

# ── gradient computation ────────────────────────────────────
grad_row = np.gradient(row_signal)
grad_col = np.gradient(col_signal)

grad_row_abs = np.abs(grad_row)
grad_col_abs = np.abs(grad_col)

# ── gradient visualization ──────────────────────────────────
fig3, axs = plt.subplots(2, 2, figsize=(10, 6))

axs[0, 0].plot(row_signal, color="blue")
axs[0, 0].set_title("Row signal")

axs[0, 1].plot(grad_row_abs, color="red")
axs[0, 1].set_title("Row gradient")

axs[1, 0].plot(col_signal, color="green")
axs[1, 0].set_title("Column signal")

axs[1, 1].plot(grad_col_abs, color="orange")
axs[1, 1].set_title("Column gradient")

for ax in axs.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "gradient_verification.png"), dpi=200)
plt.close()

print("saved gradient verification")

# ── threshold detection ─────────────────────────────────────


from scipy.signal import find_peaks

row_peaks, _ = find_peaks(row_signal, prominence=0.05, distance=15)
col_peaks, _ = find_peaks(col_signal, prominence=0.05, distance=15)




print(f"Clustered row peaks: {len(row_peaks)}")
print(f"Clustered col peaks: {len(col_peaks)}")



# ── visualization of clustered detections ───────────────────
fig4, axs = plt.subplots(2, 2, figsize=(10, 6))

axs[0, 0].plot(row_signal, color="blue")
axs[0, 0].scatter(row_peaks, row_signal[row_peaks], color="red", s=40)
axs[0, 0].set_title("Row signal + peaks")

axs[0, 1].plot(grad_row_abs, color="red")
axs[0, 1].set_title("Row gradient")

axs[1, 0].plot(col_signal, color="green")
axs[1, 0].scatter(col_peaks, col_signal[col_peaks], color="orange", s=40)
axs[1, 0].set_title("Column signal + peaks")

axs[1, 1].plot(grad_col_abs, color="orange")
axs[1, 1].set_title("Column gradient")

for ax in axs.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "clustered_detection.png"), dpi=200)
plt.close()

print("saved clustered detections")

# ── overlay on image ────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(6, 6))

ax.imshow(img_smoothed, cmap="inferno")

# row peaks
ax.scatter(row_peaks,
           np.full_like(row_peaks, mid_row_idx),
           color="red", s=30, label="row peaks")

# column peaks
ax.scatter(np.full_like(col_peaks, mid_col_idx),
           col_peaks,
           color="cyan", s=30, label="col peaks")

# slice lines
ax.axhline(mid_row_idx, color="white", linestyle="--", alpha=0.5)
ax.axvline(mid_col_idx, color="white", linestyle="--", alpha=0.5)

ax.set_title("Clustered detections on CSD")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "detections_overlay_clean.png"), dpi=200)
plt.close()

print("saved clean overlay")