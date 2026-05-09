"""
loads the extracted patches, trains an SVM classifier, evaluates it,
and saves the trained model.
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT      = "classifier_training"
PATCH_DIR = os.path.join(ROOT, "patches")
MODEL_OUT = os.path.join(ROOT, "svm_model.pkl")
FIG_OUT   = os.path.join(ROOT, "svm_confusion_matrix.png")

# load patches
print("Loading patches...")
patches = np.load(os.path.join(PATCH_DIR, "all_patches.npy")) 
labels  = np.load(os.path.join(PATCH_DIR, "all_labels.npy"))
print(f"Patches: {patches.shape}  |  Pos: {(labels==1).sum()}  Neg: {(labels==0).sum()}")

# extract HOG features instead of raw pixels
def extract_hog(patch):
    return hog(patch,
               orientations=9,          # 9 angle bins (0-180°)
               pixels_per_cell=(8, 8),  # local gradient cells
               cells_per_block=(2, 2),  # normalisation block
               visualize=False)

print("Extracting HOG features... (This forces the SVM to look at structure, not noise)")
X = np.array([extract_hog(p) for p in patches])
y = labels
print(f"HOG feature vector size: {X.shape[1]}  (Compressed from 7225 raw pixels)")

# train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# pipeline: scale -> SVM 
#  removed class_weight="balanced"
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=5.0, gamma="scale",
                   probability=True, random_state=42)),
])

print("\nTraining SVM...")
pipeline.fit(X_train, y_train)

# evaluate 
y_pred = pipeline.predict(X_test)
print("\n" + classification_report(y_test, y_pred, target_names=["negative", "positive"]))

cm = confusion_matrix(y_test, y_pred)
print(f"TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

# plot confusion matrix 
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["negative", "positive"]
)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("SVM Patch Classifier Confusion Matrix\n"
             f"Test set: {len(X_test)} patches")
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150)
plt.close()

with open(MODEL_OUT, "wb") as f:
    pickle.dump(pipeline, f)
print(f"\n✅ Smarter, structure-based model saved → {MODEL_OUT}")