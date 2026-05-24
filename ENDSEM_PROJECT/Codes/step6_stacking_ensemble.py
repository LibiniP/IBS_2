"""
STEP 6 — STACKING ENSEMBLE (FIXED)
===================================
Fixes vs original:
  1. Loads DNABERT embeddings (same feature space as Step 4)
  2. SMOTE applied inside every CV fold (consistent with Step 4)
  3. BASE path used consistently; outputs saved to BASE
  4. BiLSTM: 50 epochs + EarlyStopping (matches Step 4)
  5. Meta-learner evaluated on held-out OOF only (no leakage)
  6. Final classification report uses proper OOF predictions, not train-predict

Models:
  Base layer  → XGBoost + BiLSTM (OOF probabilities)
  Meta-learner → Logistic Regression

Outputs:
  oof_meta_features.npy
  stacking_results.txt
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU (matches Step 4)

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE          # FIX #2
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────
# BASE PATH  (FIX #3)
# ─────────────────────────────────────────────
BASE = Path(r"C:\EVERYTHING\AIE\2nd year\4th Sem\IBS-2\Project\Curated Dataset Experiments")

# ─────────────────────────────────────────────
# LOAD DATA  (FIX #1 — include DNABERT)
# ─────────────────────────────────────────────
print("\n[1/4] Loading data...")

X_manual  = np.load(BASE / "features_manual.npy")
labels_df = pd.read_csv(BASE / "labels.csv")
y_raw     = labels_df["label"].values

bert_path = BASE / "embeddings_dnabert2.npy"
if bert_path.exists():
    X_bert = np.load(bert_path)
    X      = np.hstack([X_manual, X_bert])
    print("  Using DNABERT embeddings")
else:
    X = X_manual
    print("  Using manual features only")

le = LabelEncoder()
y  = le.fit_transform(y_raw)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_classes = len(le.classes_)
print(f"  Feature shape : {X_scaled.shape}")
print(f"  Classes       : {list(le.classes_)}")
print(f"  Distribution  : {Counter(y_raw)}")

# ─────────────────────────────────────────────
# CV SETUP
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────
# BILSTM BUILDER  (FIX #4 — EarlyStopping used at call site)
# ─────────────────────────────────────────────
def build_bilstm(n_steps, feat_per_step, n_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_steps, feat_per_step)),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ─────────────────────────────────────────────
# OOF CONTAINERS
# ─────────────────────────────────────────────
oof_xgb    = np.zeros((len(y), n_classes))
oof_bilstm = np.zeros((len(y), n_classes))

n_steps       = 4
feat_per_step = X_scaled.shape[1] // n_steps
used_feats    = n_steps * feat_per_step
dropped       = X_scaled.shape[1] - used_feats
if dropped > 0:
    print(f"\n  ⚠ {dropped} features dropped from BiLSTM input "
          f"(not divisible by {n_steps} steps)")

# ─────────────────────────────────────────────
# CV LOOP — BASE MODELS
# ─────────────────────────────────────────────
print("\n[2/4] Generating Out-Of-Fold predictions...")

for fold, (tr_idx, te_idx) in enumerate(cv.split(X_scaled, y)):
    print(f"\n  Fold {fold+1}/5")

    X_tr, X_te = X_scaled[tr_idx], X_scaled[te_idx]
    y_tr, y_te = y[tr_idx],        y[te_idx]

    # ── SMOTE (FIX #2) ──────────────────────
    sm             = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
    print(f"    After SMOTE: {Counter(y_tr_res)}")

    # ── XGBoost ─────────────────────────────
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42,
        n_jobs=-1, verbosity=0
    )
    xgb.fit(X_tr_res, y_tr_res)
    oof_xgb[te_idx] = xgb.predict_proba(X_te)
    print(f"    XGBoost fold acc: "
          f"{accuracy_score(y_te, xgb.predict(X_te)):.4f}")

    # ── BiLSTM ──────────────────────────────
    X_tr_lstm = X_tr_res[:, :used_feats].reshape(-1, n_steps, feat_per_step)
    X_te_lstm = X_te[:,    :used_feats].reshape(-1, n_steps, feat_per_step)

    bilstm = build_bilstm(n_steps, feat_per_step, n_classes)

    # FIX #4 — 50 epochs + EarlyStopping (matches Step 4)
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)

    bilstm.fit(
        X_tr_lstm, y_tr_res,
        epochs=50, batch_size=32,
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )

    oof_bilstm[te_idx] = bilstm.predict(X_te_lstm, verbose=0)
    bilstm_pred        = np.argmax(oof_bilstm[te_idx], axis=1)
    print(f"    BiLSTM fold acc : "
          f"{accuracy_score(y_te, bilstm_pred):.4f}")

    keras.backend.clear_session()   # free GPU/CPU memory between folds

# ─────────────────────────────────────────────
# META FEATURES
# ─────────────────────────────────────────────
X_meta = np.hstack([oof_xgb, oof_bilstm])
print(f"\n  Meta-feature shape: {X_meta.shape}")

# ─────────────────────────────────────────────
# META-LEARNER EVALUATION  (FIX #5 — single OOF CV, no leakage)
# ─────────────────────────────────────────────
print("\n[3/4] Evaluating meta-learner (Logistic Regression)...")

oof_meta_preds = np.zeros(len(y), dtype=int)

meta_f1_scores   = []
meta_bal_acc     = []
meta_acc_scores  = []
meta_prec_scores = []
meta_rec_scores  = []

for fold, (tr_idx, te_idx) in enumerate(cv.split(X_meta, y)):
    X_tr_meta, X_te_meta = X_meta[tr_idx], X_meta[te_idx]
    y_tr_meta, y_te_meta = y[tr_idx],      y[te_idx]

    meta_lr = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42)
    meta_lr.fit(X_tr_meta, y_tr_meta)

    preds = meta_lr.predict(X_te_meta)
    oof_meta_preds[te_idx] = preds

    meta_f1_scores.append(
        f1_score(y_te_meta, preds, average="macro", zero_division=0))
    meta_bal_acc.append(
        balanced_accuracy_score(y_te_meta, preds))
    meta_acc_scores.append(
        accuracy_score(y_te_meta, preds))
    meta_prec_scores.append(
        precision_score(y_te_meta, preds, average="macro", zero_division=0))
    meta_rec_scores.append(
        recall_score(y_te_meta, preds, average="macro", zero_division=0))

# ─────────────────────────────────────────────
# FINAL RESULTS  (FIX #6 — OOF report, not train-predict)
# ─────────────────────────────────────────────
print("\n[4/4] Final Results")
print("=" * 60)
print("STACKING ENSEMBLE — OOF EVALUATION")
print("=" * 60)
print(f"  Accuracy          : {np.mean(meta_acc_scores):.4f} "
      f"± {np.std(meta_acc_scores):.4f}")
print(f"  Balanced Accuracy : {np.mean(meta_bal_acc):.4f} "
      f"± {np.std(meta_bal_acc):.4f}")
print(f"  Macro F1          : {np.mean(meta_f1_scores):.4f} "
      f"± {np.std(meta_f1_scores):.4f}")
print(f"  Macro Precision   : {np.mean(meta_prec_scores):.4f} "
      f"± {np.std(meta_prec_scores):.4f}")
print(f"  Macro Recall      : {np.mean(meta_rec_scores):.4f} "
      f"± {np.std(meta_rec_scores):.4f}")

print("\nClassification Report (aggregated OOF — unbiased):\n")
print(classification_report(
    y, oof_meta_preds,
    target_names=le.classes_,
    zero_division=0
))

print("Confusion Matrix (OOF):")
cm = confusion_matrix(y, oof_meta_preds)
print(pd.DataFrame(
    cm,
    index  =[f"True_{c}"  for c in le.classes_],
    columns=[f"Pred_{c}" for c in le.classes_]
))

# ─────────────────────────────────────────────
# SAVE OUTPUTS  (FIX #3 — save to BASE)
# ─────────────────────────────────────────────
np.save(BASE / "oof_meta_features.npy", X_meta)
print(f"\nSaved → {BASE / 'oof_meta_features.npy'}")

with open(BASE / "stacking_results.txt", "w") as f:
    f.write("STACKING ENSEMBLE RESULTS (OOF — unbiased)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Accuracy          : {np.mean(meta_acc_scores):.4f} "
            f"± {np.std(meta_acc_scores):.4f}\n")
    f.write(f"Balanced Accuracy : {np.mean(meta_bal_acc):.4f} "
            f"± {np.std(meta_bal_acc):.4f}\n")
    f.write(f"Macro F1          : {np.mean(meta_f1_scores):.4f} "
            f"± {np.std(meta_f1_scores):.4f}\n")
    f.write(f"Macro Precision   : {np.mean(meta_prec_scores):.4f} "
            f"± {np.std(meta_prec_scores):.4f}\n")
    f.write(f"Macro Recall      : {np.mean(meta_rec_scores):.4f} "
            f"± {np.std(meta_rec_scores):.4f}\n\n")
    f.write("Classification Report (OOF):\n")
    f.write(classification_report(
        y, oof_meta_preds,
        target_names=le.classes_,
        zero_division=0
    ))
    f.write("\nConfusion Matrix (OOF):\n")
    f.write(str(pd.DataFrame(
        cm,
        index  =[f"True_{c}"  for c in le.classes_],
        columns=[f"Pred_{c}" for c in le.classes_]
    )))

print(f"Saved → {BASE / 'stacking_results.txt'}")
print("\n================ DONE ================")