"""
loads the extracted patches, trains an SVM classifier, evaluates it,
and saves the trained model.

in:
  all_patches.npy   
  all_labels.npy    

out:
  svm_model.pkl             trained model
  svm_confusion_matrix.png  evaluation figure
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay)

#  config 
ROOT      = "classifier_training"
PATCH_DIR = os.path.join(ROOT, "patches")
MODEL_OUT = os.path.join(ROOT, "svm_model.pkl")
FIG_OUT   = os.path.join(ROOT, "svm_confusion_matrix.png")

#  load data 
print("Loading patches...")
patches = np.load(os.path.join(PATCH_DIR, "all_patches.npy"))  # (400, 85, 85)
labels  = np.load(os.path.join(PATCH_DIR, "all_labels.npy"))   # (400,)

print(f"  Patches shape : {patches.shape}")
print(f"  Labels shape  : {labels.shape}")
print(f"  Positives     : {(labels == 1).sum()}")
print(f"  Negatives     : {(labels == 0).sum()}")

# flatten patches ---> feature vectors 
# each 85x85 patch becomes a 7225-element vector
N = patches.shape[0]
X = patches.reshape(N, -1).astype(np.float32)  # (400, 7225)
y = labels

print(f"\n  Feature matrix shape: {X.shape}")

#  train / test split 
# both classes are represented equally in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\n  Train samples : {len(X_train)}  "
      f"(pos={y_train.sum()}, neg={(y_train==0).sum()})")
print(f"  Test  samples : {len(X_test)}   "
      f"(pos={y_test.sum()},  neg={(y_test==0).sum()})")

#  build pipeline: scale → SVM 
# StandardScaler normalises each pixel position across all patches
# class_weight="balanced" compensates for 1:3 pos:neg ratio
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(
        kernel="rbf",           # radial basis — best for image patches
        C=10.0,                 # regularisation strength
        gamma="scale",          # auto-scale to feature variance
        class_weight="balanced",# compensate for class imbalance
        random_state=42,
        probability=True,       # enable predict_proba for the navigator
    ))
])

print("\nTraining SVM... (this may take ~1-2 minutes on 320 samples)")
pipeline.fit(X_train, y_train)
print("Training complete.")

#  evaluate 
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]   # probability of positive class

print("\n" + "─"*50)
print("CLASSIFICATION REPORT")
print("─"*50)
print(classification_report(y_test, y_pred,
                             target_names=["negative (no cell)",
                                           "positive (cell interior)"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

#  plot confusion matrix 
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
print(f"\nConfusion matrix saved → {FIG_OUT}")

#  save model 
with open(MODEL_OUT, "wb") as f:
    pickle.dump(pipeline, f)

print(f"Model saved       → {MODEL_OUT}")
print("\nDone. The navigator can now load svm_model.pkl to classify patches.")