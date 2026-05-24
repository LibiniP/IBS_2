"""
STEP 4 — FULL TRAINING PIPELINE (FIXED)
Fixes:
- RF: added SMOTE like XGBoost
- CNN: added SMOTE + more epochs + better architecture
- BiLSTM: added SMOTE
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# BASE PATH
# ─────────────────────────────────────────────
BASE = Path(r"C:\EVERYTHING\AIE\2nd year\4th Sem\IBS-2\Project\Curated Dataset Experiments")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
labels_df = pd.read_csv(BASE / "labels.csv")
y_raw     = labels_df["label"].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y  = le.fit_transform(y_raw)

print("Classes:", le.classes_)
print("Distribution:", Counter(y_raw))

X_manual  = np.load(BASE / "features_manual.npy")
bert_path = BASE / "embeddings_dnabert2.npy"
if bert_path.exists():
    X_bert = np.load(bert_path)
    X = np.hstack([X_manual, X_bert])
    print("Using DNABERT embeddings")
else:
    X = X_manual
    print("Using manual features only")
print("Feature shape:", X.shape)

# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────
from sklearn.utils.class_weight import compute_class_weight
classes           = np.unique(y)
class_weights     = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

# ─────────────────────────────────────────────
# CV SETUP
# ─────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    roc_auc_score
)

def compute_all_metrics(y_true, y_pred, y_prob=None):
    cm        = confusion_matrix(y_true, y_pred)
    n_classes = len(np.unique(y_true))

    sensitivities, specificities = [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    metrics = {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1":          round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1":       round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "macro_precision":   round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_recall":      round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "mean_sensitivity":  round(np.mean(sensitivities), 4),
        "mean_specificity":  round(np.mean(specificities), 4),
        "confusion_matrix":  cm,
    }

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, cls in enumerate(le.classes_):
        metrics[f"f1_{cls}"]          = round(per_class_f1[i], 4)
        metrics[f"sensitivity_{cls}"] = round(sensitivities[i], 4)
        metrics[f"specificity_{cls}"] = round(specificities[i], 4)

    if y_prob is not None:
        try:
            if n_classes == 2:
                metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob[:, 1]), 4)
            else:
                metrics["roc_auc"] = round(roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"), 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def aggregate_fold_metrics(fold_metrics_list):
    keys = [k for k in fold_metrics_list[0].keys() if k != "confusion_matrix"]
    agg  = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics_list if m[k] is not None]
        agg[k]          = round(np.mean(vals), 4) if vals else None
        agg[f"{k}_std"] = round(np.std(vals),  4) if vals else None
    agg["confusion_matrix"] = sum(m["confusion_matrix"] for m in fold_metrics_list)
    return agg


def print_results(name, res):
    print(f"  Accuracy         : {res['accuracy']}")
    print(f"  Balanced Accuracy: {res['balanced_accuracy']}")
    print(f"  Macro F1         : {res['macro_f1']}")
    print(f"  Mean Sensitivity : {res['mean_sensitivity']}")
    print(f"  Mean Specificity : {res['mean_specificity']}")
    print(f"  ROC-AUC          : {res['roc_auc']}")
    print(f"  Confusion Matrix :\n{res['confusion_matrix']}")


results = {}

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# MODEL 1 — RANDOM FOREST + SMOTE ✅ FIXED
# ─────────────────────────────────────────────
print("\nMODEL 1: Random Forest + SMOTE")

rf_fold_metrics = []

for tr, te in cv.split(X, y):
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X[tr])
    X_te   = scaler.transform(X[te])

    # ✅ SMOTE inside CV — same as XGBoost
    sm             = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y[tr])

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1)
    rf.fit(X_tr_res, y_tr_res)

    pred  = rf.predict(X_te)
    proba = rf.predict_proba(X_te)
    rf_fold_metrics.append(compute_all_metrics(y[te], pred, proba))

results["RandomForest"] = aggregate_fold_metrics(rf_fold_metrics)
print_results("RandomForest", results["RandomForest"])

# ─────────────────────────────────────────────
# MODEL 2 — XGBOOST + SMOTE (unchanged, working)
# ─────────────────────────────────────────────
print("\nMODEL 2: XGBoost + SMOTE")

xgb_fold_metrics = []

for tr, te in cv.split(X, y):
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X[tr])
    X_te   = scaler.transform(X[te])

    sm             = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y[tr])

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, n_jobs=-1)
    xgb.fit(X_tr_res, y_tr_res)

    pred  = xgb.predict(X_te)
    proba = xgb.predict_proba(X_te)
    xgb_fold_metrics.append(compute_all_metrics(y[te], pred, proba))

results["XGBoost"] = aggregate_fold_metrics(xgb_fold_metrics)
print_results("XGBoost", results["XGBoost"])

# ─────────────────────────────────────────────
# MODEL 3 — CNN + SMOTE ✅ FIXED
# ─────────────────────────────────────────────
print("\nMODEL 3: CNN + SMOTE")

try:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ✅ force CPU — GPU broken on Windows
    import tensorflow as tf
    from tensorflow import keras

    n_classes        = len(le.classes_)
    cnn_fold_metrics = []

    for tr, te in cv.split(X, y):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[tr])
        X_te   = scaler.transform(X[te])

        # ✅ SMOTE before reshape
        sm             = SMOTE(random_state=42)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y[tr])

        X_tr_res = X_tr_res.reshape(X_tr_res.shape[0], X_tr_res.shape[1], 1)
        X_te_res = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)

        # ✅ Better architecture with BatchNorm
        model = keras.Sequential([
            keras.layers.Conv1D(64, 5, activation="relu",
                                input_shape=(X_tr_res.shape[1], 1)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(128, 3, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(n_classes, activation="softmax")
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

        # ✅ More epochs + early stopping
        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True)

        model.fit(X_tr_res, y_tr_res,
                  epochs=50, batch_size=32,
                  validation_split=0.1,
                  callbacks=[es],
                  verbose=0)

        pred  = np.argmax(model.predict(X_te_res, verbose=0), axis=1)
        proba = model.predict(X_te_res, verbose=0)
        cnn_fold_metrics.append(compute_all_metrics(y[te], pred, proba))

    results["CNN"] = aggregate_fold_metrics(cnn_fold_metrics)
    print_results("CNN", results["CNN"])

except Exception as e:
    print("CNN skipped:", e)
    results["CNN"] = {"balanced_accuracy": None, "macro_f1": None,
                      "confusion_matrix": None}

# ─────────────────────────────────────────────
# MODEL 4 — BiLSTM + SMOTE ✅ FIXED
# ─────────────────────────────────────────────
print("\nMODEL 4: BiLSTM + SMOTE")

try:
    from tensorflow import keras

    n_steps           = 4
    lstm_fold_metrics = []

    for tr, te in cv.split(X, y):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[tr])
        X_te   = scaler.transform(X[te])

        # ✅ SMOTE before reshape
        sm             = SMOTE(random_state=42)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y[tr])

        feat_per_step = X_tr_res.shape[1] // n_steps
        X_tr_res = X_tr_res[:, :n_steps*feat_per_step].reshape(
            -1, n_steps, feat_per_step)
        X_te_res = X_te[:, :n_steps*feat_per_step].reshape(
            -1, n_steps, feat_per_step)

        model = keras.Sequential([
            keras.layers.Bidirectional(
                keras.layers.LSTM(64, return_sequences=True),
                input_shape=(n_steps, feat_per_step)),
            keras.layers.Bidirectional(keras.layers.LSTM(32)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(len(le.classes_), activation="softmax")
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

        # ✅ Early stopping
        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True)

        model.fit(X_tr_res, y_tr_res,
                  epochs=50, batch_size=32,
                  validation_split=0.1,
                  callbacks=[es],
                  verbose=0)

        pred  = np.argmax(model.predict(X_te_res, verbose=0), axis=1)
        proba = model.predict(X_te_res, verbose=0)
        lstm_fold_metrics.append(compute_all_metrics(y[te], pred, proba))

    results["BiLSTM"] = aggregate_fold_metrics(lstm_fold_metrics)
    print_results("BiLSTM", results["BiLSTM"])

except Exception as e:
    print("BiLSTM skipped:", e)
    results["BiLSTM"] = {"balanced_accuracy": None, "macro_f1": None,
                         "confusion_matrix": None}

# ─────────────────────────────────────────────
# FINAL RESULTS TABLE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

summary_cols = [
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
    "macro_precision", "macro_recall",
    "mean_sensitivity", "mean_specificity", "roc_auc"
]

rows = []
for model_name, metrics in results.items():
    row = {"model": model_name}
    for col in summary_cols:
        row[col] = metrics.get(col, None)
    rows.append(row)

results_df = pd.DataFrame(rows).set_index("model")
print(results_df.to_string())

results_df.to_csv(BASE / "results_summary.csv")
print("\nSaved → results_summary.csv")

print("\nConfusion Matrices:")
for model_name, metrics in results.items():
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        print(f"\n{model_name}:")
        print(pd.DataFrame(cm,
                           index=[f"True_{c}" for c in le.classes_],
                           columns=[f"Pred_{c}" for c in le.classes_]))
        pd.DataFrame(cm).to_csv(BASE / f"cm_{model_name}.csv", index=False)