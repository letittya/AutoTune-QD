# AutoTune-QD

Automated crosstalk extraction and virtual gate recovery in double quantum dot (DQD) charge stability diagrams (CSDs).

Starting from a single CSD image and no prior knowledge of the device, the system extracts the crosstalk coefficients α₁₂ and α₂₁, recovers the virtual gate matrix, reports a 95% confidence interval for each coefficient, and validates the result with an orthogonality check. Two independent extraction pipelines (1D slice-based and 2D Hough-based) are implemented and compared. A navigator demonstrates the pipeline end-to-end by locating the (1,1) charge state on unseen images.

## Overview

The project is organized around four stages:

1. **Data generation** — CSDs are simulated with two independent frameworks, QArray and SimCATS, with controllable crosstalk and noise.
2. **Extraction** — two pipelines recover the transition-line slopes:
   - **1D pipeline**: image slicing → peak detection → two-pass iterative RANSAC → MAD filtering.
   - **2D pipeline**: Canny edge detection → probabilistic Hough transform → segment clustering → RANSAC re-fitting → MAD filtering.
3. **Virtual gate recovery** — slopes are converted into the crosstalk coefficients and the virtual gate matrix, with 95% confidence intervals (Student's t-distribution) and an orthogonality validation.
4. **Navigator** — uses the 1D pipeline plus an SVM patch classifier to localize the (1,1) charge state on new images.

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `matplotlib`
- [QArray](https://github.com/b-vanstraaten/qarray)
- [SimCATS](https://github.com/f-hader/SimCATS)

Install the core dependencies with:

```bash
pip install numpy scipy scikit-image scikit-learn matplotlib
```

Then install QArray and SimCATS following the instructions in their respective repositories.

## Usage

**1. Generate CSDs**

```bash
python CSD_gen_QArray.py
python CSD_gen_simCAT.py
```

**2. Run the extraction pipelines**

```bash
python 1D/1D_extraction.py
python 2D/2D_extraction.py
```

Each produces an `extracted_lines.json` with the detected line slopes and the recovered virtual gate matrix.

**3. Train the SVM classifier**

```bash
python classifier_training/generate_training_data.py
python classifier_training/extract_patches.py
python classifier_training/svm_trainer.py
```

**4. Run the navigator**

Open `navigator/navigator_code.ipynb` and run the cells, or run the navigator script on a test image to locate the (1,1) state.


## Results

On a controlled benchmark sweeping crosstalk strength (α) and spatial distortion (wobble) independently, the 1D pipeline achieved a mean error of ≈9.4% and the 2D pipeline ≈10.9%, with the 1D pipeline being more robust to spatial distortion. The navigator localized the (1,1) state on unseen images with a median error of ≈1 pixel.

