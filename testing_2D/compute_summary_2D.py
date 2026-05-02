"""
compile_2D_summary.py
─────────────
Scans all 12 test folders in testing_2D/results/, extracts the physics matrices 
and line counts, and compiles them into a single CSV for your thesis tables.
"""

import os
import json
import csv

RESULTS_DIR = os.path.join("testing_2D", "results")

# Find all test subfolders (e.g. "1_alpha_0.10")
folders = sorted([f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))])

summary_data = []

for folder_name in folders:
    # Parse ground truth from folder name 
    parts = folder_name.split("_")
    if len(parts) < 3:
        continue
        
    try:
        idx = int(parts[0])
        test_type = parts[1]
        
        # Handle the .png extension in the folder name if it's there
        val_str = parts[2].replace(".png", "")
        val = float(val_str)
    except ValueError:
        continue
    
    # Reconstruct what the physics simulator used
    gt_alpha = val if test_type == "alpha" else 0.25
    wobble = val if test_type == "wobble" else 35.0

    # Paths to the specific JSONs for this test
    base_dir = os.path.join(RESULTS_DIR, folder_name)
    vg_json_path = os.path.join(base_dir, "Virtual_Gates", "vg_matrix.json")
    lines_json_path = os.path.join(base_dir, "extracted_lines.json")

    if not os.path.exists(vg_json_path) or not os.path.exists(lines_json_path):
        print(f"Skipping {folder_name} (missing JSONs)")
        continue

    # Load the data
    with open(vg_json_path, "r") as f:
        vg_data = json.load(f)
        
    with open(lines_json_path, "r") as f:
        lines_data = json.load(f)

    # Count lines that survived the MAD filter
    n_diag = sum(1 for l in lines_data if l["type"] == "diagonal")
    n_steep = sum(1 for l in lines_data if l["type"] == "steep")

    # Extract physics metrics
    a12 = vg_data["M_physics"][0][1]
    a21 = vg_data["M_physics"][1][0]
    
    ci_diag = vg_data["diagonal_95_CI"]
    ci_steep = vg_data["steep_95_CI"]
    
    # Calculate widths and absolute percentage errors
    a12_err = 100 * abs(a12 - gt_alpha) / gt_alpha
    a21_err = 100 * abs(a21 - gt_alpha) / gt_alpha
    
    ci_diag_width = ci_diag[1] - ci_diag[0]
    ci_steep_width = ci_steep[1] - ci_steep[0]

    summary_data.append({
        "ID": idx,
        "Test_Name": folder_name,
        "GT_Alpha": gt_alpha,
        "Wobble_px": wobble,
        "N_Diag": n_diag,
        "N_Steep": n_steep,
        "a12_Meas": round(a12, 4),
        "a12_Err_%": round(a12_err, 2),
        "a12_CI_Width": round(ci_diag_width, 4),
        "a21_Meas": round(a21, 4),
        "a21_Err_%": round(a21_err, 2),
        "a21_CI_Width": round(ci_steep_width, 4),
        "Orthogonality_Deg": round(vg_data["orthogonality_angle_deg"], 2)
    })

# Sort numerically by Test ID
summary_data.sort(key=lambda x: x["ID"])

# Write everything to a clean CSV
csv_path = os.path.join(RESULTS_DIR, "benchmark_summary_2D.csv")
if summary_data:
    keys = summary_data[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"\nSUCCESS! Aggregated {len(summary_data)} 2D tests.")
    print(f"Summary saved to -> {csv_path}")
else:
    print("No data found to summarize.")