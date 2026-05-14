"""
1D (peak detection + RANSAC) vs 2D (Canny + Hough + RANSAC) 

used benchmark_summary and benchmark_summary_2D:
  -alpha error vs ground-truth alpha 
  -alpha error vs wobble amplitude 
  -number of detected lines vs alpha and wobble
  -95% CI width vs alpha and wobble
  -summary table image
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_1D = os.path.join(HERE, "..", "testing_1D", "results", "benchmark_summary.csv")
CSV_2D = os.path.join(HERE, "..", "testing_2D", "results", "benchmark_summary_2D.csv")

df1 = pd.read_csv(CSV_1D)
df2 = pd.read_csv(CSV_2D)

# helpers

def mean_alpha_err(df):
    """avrg of a12 and a21 percentage errors."""
    return (df["a12_Err_%"] + df["a21_Err_%"]) / 2

def mean_ci(df):
    return (df["a12_CI_Width"] + df["a21_CI_Width"]) / 2

df1["mean_err"] = mean_alpha_err(df1)
df2["mean_err"] = mean_alpha_err(df2)
df1["mean_ci"]  = mean_ci(df1)
df2["mean_ci"]  = mean_ci(df2)

alpha_sweep  = df1[df1["ID"] <= 6].copy()
alpha_sweep2 = df2[df2["ID"] <= 6].copy()
wobble_sweep  = df1[df1["ID"] >= 7].copy()
wobble_sweep2 = df2[df2["ID"] >= 7].copy()

C1, C2 = "#2563EB", "#DC2626"   # blue = 1D, red = 2D
MS = 7

#1 mean alpha error vs ground-truth alpha 

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
fig.suptitle("Alpha Sweep  (wobble = 35 px fixed)", fontsize=13, fontweight="bold")

for ax, (col, label) in zip(axes, [("a12_Err_%", "α₁₂"), ("a21_Err_%", "α₂₁")]):
    ax.plot(alpha_sweep["GT_Alpha"],  alpha_sweep[col],  "o-", color=C1, ms=MS, label="1D")
    ax.plot(alpha_sweep2["GT_Alpha"], alpha_sweep2[col], "s--", color=C2, ms=MS, label="2D")
    ax.set_xlabel("Ground-truth α", fontsize=11)
    ax.set_ylabel("Absolute error (%)", fontsize=11)
    ax.set_title(f"{label} error", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(alpha_sweep["GT_Alpha"])

plt.tight_layout()
fig.savefig(os.path.join(HERE, "fig1_alpha_sweep_error.png"), dpi=150)
plt.close(fig)
print("Saved fig1_alpha_sweep_error.png")

#2 mean alpha error vs wobble 

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Wobble Sweep  (α = 0.25 fixed)", fontsize=13, fontweight="bold")

for ax, (col, label) in zip(axes, [("a12_Err_%", "α₁₂"), ("a21_Err_%", "α₂₁")]):
    ax.plot(wobble_sweep["Wobble_px"],  wobble_sweep[col],  "o-",  color=C1, ms=MS, label="1D")
    ax.plot(wobble_sweep2["Wobble_px"], wobble_sweep2[col], "s--", color=C2, ms=MS, label="2D")
    ax.set_xlabel("Wobble amplitude (px)", fontsize=11)
    ax.set_ylabel("Absolute error (%)", fontsize=11)
    ax.set_title(f"{label} error vs wobble", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(wobble_sweep["Wobble_px"])

plt.tight_layout()
fig.savefig(os.path.join(HERE, "fig2_wobble_sweep_error.png"), dpi=150)
plt.close(fig)
print("Saved fig2_wobble_sweep_error.png")

#3 number of detected lines 

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Number of Detected Lines: 1D vs 2D", fontsize=13, fontweight="bold")

pairs = [
    (alpha_sweep,  alpha_sweep2,  "GT_Alpha",  "Ground-truth α",       "Alpha sweep"),
    (wobble_sweep, wobble_sweep2, "Wobble_px", "Wobble amplitude (px)", "Wobble sweep"),
]
line_types = [("N_Diag", "Diagonal lines"), ("N_Steep", "Steep lines")]

for col_idx, (lt_col, lt_label) in enumerate(line_types):
    for row_idx, (sw1, sw2, x_col, x_label, title) in enumerate(pairs):
        ax = axes[row_idx][col_idx]
        ax.plot(sw1[x_col], sw1[lt_col], "o-",  color=C1, ms=MS, label="1D")
        ax.plot(sw2[x_col], sw2[lt_col], "s--", color=C2, ms=MS, label="2D")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{lt_label} — {title}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(HERE, "fig3_line_counts.png"), dpi=150)
plt.close(fig)
print("Saved fig3_line_counts.png")

#4 95% CI width 

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("95% Confidence Interval Width on α", fontsize=13, fontweight="bold")

for ax, (sw1, sw2, x_col, x_label, title) in zip(
    axes,
    [
        (alpha_sweep,  alpha_sweep2,  "GT_Alpha",  "Ground-truth α",       "Alpha sweep (wobble=35px)"),
        (wobble_sweep, wobble_sweep2, "Wobble_px", "Wobble amplitude (px)", "Wobble sweep (α=0.25)"),
    ],
):
    ax.plot(sw1[x_col], sw1["mean_ci"], "o-",  color=C1, ms=MS, label="1D")
    ax.plot(sw2[x_col], sw2["mean_ci"], "s--", color=C2, ms=MS, label="2D")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Mean CI width", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(HERE, "fig4_ci_width.png"), dpi=150)
plt.close(fig)
print("Saved fig4_ci_width.png")

#5 side-by-side summary table

cols_show = ["Test_Name", "GT_Alpha", "Wobble_px",
             "a12_Err_%", "a21_Err_%", "mean_err"]

merged = df1[cols_show].copy().rename(columns={
    "a12_Err_%": "1D a12 err%",
    "a21_Err_%": "1D a21 err%",
    "mean_err":  "1D mean err%",
})
merged["2D a12 err%"]  = df2["a12_Err_%"].values
merged["2D a21 err%"]  = df2["a21_Err_%"].values
merged["2D mean err%"] = df2["mean_err"].values
merged["Δ mean err%"]  = (df2["mean_err"] - df1["mean_err"]).round(2)  # positive = 2D worse

display_cols = ["Test_Name", "GT_Alpha", "Wobble_px",
                "1D mean err%", "2D mean err%", "Δ mean err%"]
tdata = merged[display_cols].round(2)

fig, ax = plt.subplots(figsize=(13, 5))
ax.axis("off")
tbl = ax.table(
    cellText=tdata.values,
    colLabels=tdata.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F0F4FF")
    # colourdelta column: green if 1D better (delta > 0), red if 2D better (delta < 0)
    if c == 5 and r > 0:
        val = tdata.iloc[r - 1]["Δ mean err%"]
        cell.set_facecolor("#D1FAE5" if val > 0 else "#FEE2E2" if val < 0 else "white")

fig.suptitle("Summary Table: 1D vs 2D Mean Alpha Error (%) — Delta = 2D − 1D",
             fontsize=11, fontweight="bold", y=0.97)
plt.tight_layout()
fig.savefig(os.path.join(HERE, "fig5_summary_table.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig5_summary_table.png")

# summary

print("\n" + "=" * 60)
print("PIPELINE COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Test':<22} {'1D err%':>9} {'2D err%':>9} {'d(2D-1D)':>11}")
print("-" * 55)
for _, row in merged.iterrows():
    delta = row["Δ mean err%"]
    flag  = "  << 2D worse" if delta > 1 else ("  << 2D better" if delta < -1 else "")
    print(f"{row['Test_Name']:<22} {row['1D mean err%']:>9.2f} {row['2D mean err%']:>9.2f} {delta:>+11.2f}{flag}")

print("=" * 60)
overall_1d = df1["mean_err"].mean()
overall_2d = df2["mean_err"].mean()
print(f"Overall mean error  —  1D: {overall_1d:.2f}%   2D: {overall_2d:.2f}%")
print(f"1D is {'better' if overall_1d < overall_2d else 'worse'} overall by {abs(overall_1d - overall_2d):.2f} percentage points")
