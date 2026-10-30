#!/usr/bin/env python3
"""
scripts/train_v3_quad_ensemble.py

Enhanced Quad-Model Stacking Ensemble Architecture:
1. Deep Recurrent Neural Network: Calibrated Bi-LSTM
2. Non-linear Kernel Support Vector Machine: Tuned RBF SVC
3. Tree-Based Ensemble: Tuned Random Forest Classifier (Captures non-linear thresholds & interactions)
4. Linear Statistical Odds: Balanced Regularized Logistic Regression
5. Out-of-Fold (OOF) 5-Fold Stratified Stacking Meta-Classifier
6. Probability Calibration: Sigmoid (Platt) vs Isotonic Calibration (min Brier & ECE)
7. Full diagnostic curve generation (ROC, PR, Calibration, Confusion Matrix, Feature Importance)
8. Complete serialization of all v3.0.0-calibrated artifacts
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score
)

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
DATA_PATH = "Wilson_disease_dataset_v2.csv"
OUT_DIR = "wilson_artifacts"
RANDOM_STATE = 42
TARGET_COL = "Is_Wilson_Disease"
MODEL_VERSION = "v3.0.0-calibrated"

os.makedirs(OUT_DIR, exist_ok=True)

# Deterministic reproducibility seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def build_enhanced_bilstm(input_timesteps: int) -> tf.keras.Model:
    """
    Constructs an enhanced, stabilized Bidirectional LSTM architecture for clinical feature sequences.
    Incorporates BatchNormalization and Dropout to regularize feature activations.
    """
    model = Sequential([
        Input(shape=(input_timesteps, 1)),
        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        mask = (bin_indices == b)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += np.abs(bin_acc - bin_conf) * (np.sum(mask) / n)
    return float(ece)


def evaluate_metrics(y_true, y_prob, threshold=0.5):
    """Computes full suite of classification and calibration metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity calculation
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "log_loss": float(ll),
        "ece": float(ece)
    }


def bootstrap_ci(y_true, y_prob, n_bootstraps=500, seed=42):
    """Calculates 95% Bootstrap Confidence Intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    auc_list, brier_list, ece_list = [], [], []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        yt_b = y_true[idx]
        yp_b = y_prob[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        auc_list.append(roc_auc_score(yt_b, yp_b))
        brier_list.append(brier_score_loss(yt_b, yp_b))
        ece_list.append(compute_ece(yt_b, yp_b))

    return {
        "roc_auc_ci95": [float(np.percentile(auc_list, 2.5)), float(np.percentile(auc_list, 97.5))],
        "brier_score_ci95": [float(np.percentile(brier_list, 2.5)), float(np.percentile(brier_list, 97.5))],
        "ece_ci95": [float(np.percentile(ece_list, 2.5)), float(np.percentile(ece_list, 97.5))]
    }


def run_training_pipeline():
    print("=" * 80)
    print(f"STARTING QUAD-MODEL STACKING ENSEMBLE TRAINING ({MODEL_VERSION})")
    print("=" * 80)

    # 1. Load dataset
    print(f"\n[STEP 1] Loading clinical dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    if "Name" in df.columns:
        df = df.drop(columns=["Name"])
    print(f"Loaded dataset shape: {df.shape} (N={len(df):,} patients, {df.shape[1]-1} features)")

    X_raw = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL].astype(int)

    # Subsample to 20,000 for balanced execution speed and robust convergence
    if len(df) > 20000:
        print(f"Subsampling {len(df):,} down to 20,000 stratified samples for balanced training...")
        X_sub, _, y_sub, _ = train_test_split(
            X_raw, y_raw, train_size=20000, stratify=y_raw, random_state=RANDOM_STATE
        )
    else:
        X_sub, y_sub = X_raw, y_raw

    # 2. Strict Pre-Split Partitioning (Zero Test Leakage)
    print("\n[STEP 2] Partitioning into Train (60%), Validation (20%), and Test (20%)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.20, stratify=y_sub, random_state=RANDOM_STATE
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
    )
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"Split sizes: Train={len(X_train_raw):,}, Val={len(X_val_raw):,}, Test={len(X_test_raw):,}")

    # 3. Fit Preprocessing exclusively on X_train_raw
    print("\n[STEP 3] Fitting Preprocessor strictly on Train split...")
    categorical_features = ["Sex", "Region", "Socioeconomic Status", "Alcohol Use"]
    numeric_features = [c for c in X_train_raw.columns if c not in categorical_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    preprocessor.fit(X_train_raw)
    preprocessor._name_to_fitted_passthrough = {}

    X_train_proc = preprocessor.transform(X_train_raw)
    X_val_proc = preprocessor.transform(X_val_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    timesteps = X_train_proc.shape[1]
    print(f"Preprocessed feature space: {timesteps} features.")

    # 4. Out-of-Fold (OOF) Stacking Pipeline on Train Split
    print("\n[STEP 4] Generating 5-Fold Stratified Out-of-Fold Predictions for 4 Base Models...")
    print("  Models: [1] Bi-LSTM  |  [2] SVM (RBF)  |  [3] Random Forest  |  [4] Logistic Regression")
    
    n_train = len(X_train_proc)
    oof_bilstm = np.zeros(n_train)
    oof_svm = np.zeros(n_train)
    oof_rf = np.zeros(n_train)
    oof_log = np.zeros(n_train)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train)):
        print(f"  --> Processing Fold {fold + 1}/5 (train={len(tr_idx)}, val={len(val_idx)})...")
        X_tr, y_tr = X_train_proc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train_proc[val_idx], y_train.iloc[val_idx]

        # A. Bi-LSTM Fold
        X_tr_lstm = X_tr.reshape((-1, timesteps, 1))
        X_va_lstm = X_va.reshape((-1, timesteps, 1))
        m_bilstm = build_enhanced_bilstm(timesteps)
        es_fold = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        m_bilstm.fit(
            X_tr_lstm, y_tr,
            validation_data=(X_va_lstm, y_va),
            epochs=15, batch_size=64,
            callbacks=[es_fold], verbose=0
        )
        oof_bilstm[val_idx] = m_bilstm.predict(X_va_lstm, verbose=0).ravel()

        # B. SVM Fold (RBF Kernel, C=1.5)
        m_svm = SVC(kernel="rbf", C=1.5, probability=True, class_weight="balanced", random_state=RANDOM_STATE)
        m_svm.fit(X_tr, y_tr)
        oof_svm[val_idx] = m_svm.predict_proba(X_va)[:, 1]

        # C. Random Forest Fold (150 trees, max_depth=12)
        m_rf = RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        m_rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = m_rf.predict_proba(X_va)[:, 1]

        # D. Logistic Regression Fold
        m_log = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        m_log.fit(X_tr, y_tr)
        oof_log[val_idx] = m_log.predict_proba(X_va)[:, 1]

    # Stacking matrix ordered as: [Bi-LSTM, SVM, Random Forest, LogReg]
    oof_stack_train = np.column_stack([oof_bilstm, oof_svm, oof_rf, oof_log])
    print(f"OOF Stack Matrix shape: {oof_stack_train.shape} (4 constituent models)")

    # 5. Train Stacking Meta-Classifier exclusively on OOF Predictions
    print("\n[STEP 5] Training Stacking Meta-Classifier on OOF Predictions...")
    meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    meta_model.fit(oof_stack_train, y_train)
    print(f"Meta-Classifier feature weights [Bi-LSTM, SVM, RF, LogReg]: {meta_model.coef_[0]}")
    print(f"Meta-Classifier Intercept: {meta_model.intercept_[0]:.4f}")

    # 6. Retrain all Base Models on Full Training Split
    print("\n[STEP 6] Retraining base models on full training split (N=12,000)...")

    # A. Bi-LSTM Final
    print("  Fitting final Bi-LSTM model...")
    bilstm_final = build_enhanced_bilstm(timesteps)
    es_final = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    rlr_final = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    X_train_lstm = X_train_proc.reshape((-1, timesteps, 1))
    X_val_lstm = X_val_proc.reshape((-1, timesteps, 1))
    bilstm_history = bilstm_final.fit(
        X_train_lstm, y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=30, batch_size=64,
        callbacks=[es_final, rlr_final], verbose=1
    )

    # B. SVM Final
    print("  Fitting final SVM (RBF Kernel, C=1.5)...")
    svm_final = SVC(kernel="rbf", C=1.5, probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    svm_final.fit(X_train_proc, y_train)

    # C. Random Forest Final
    print("  Fitting final Random Forest (150 trees, max_depth=12)...")
    rf_final = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_final.fit(X_train_proc, y_train)

    # D. Logistic Regression Final
    print("  Fitting final Logistic Regression...")
    logreg_final = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    logreg_final.fit(X_train_proc, y_train)

    # 7. Validation Split Evaluation & Probability Calibration
    print("\n[STEP 7] Evaluating Validation Split and Fitting Calibrators...")
    val_bilstm_p = bilstm_final.predict(X_val_lstm, verbose=0).ravel()
    val_svm_p = svm_final.predict_proba(X_val_proc)[:, 1]
    val_rf_p = rf_final.predict_proba(X_val_proc)[:, 1]
    val_log_p = logreg_final.predict_proba(X_val_proc)[:, 1]
    val_stack = np.column_stack([val_bilstm_p, val_svm_p, val_rf_p, val_log_p])

    val_meta_raw_p = meta_model.predict_proba(val_stack)[:, 1]
    metrics_val_raw = evaluate_metrics(y_val.to_numpy(), val_meta_raw_p)

    # Sigmoid vs Isotonic Calibrators
    calibrator_sigmoid = CalibratedClassifierCV(estimator=meta_model, method="sigmoid", cv="prefit")
    calibrator_sigmoid.fit(val_stack, y_val)
    val_prob_sig = calibrator_sigmoid.predict_proba(val_stack)[:, 1]
    metrics_val_sig = evaluate_metrics(y_val.to_numpy(), val_prob_sig)

    calibrator_isotonic = CalibratedClassifierCV(estimator=meta_model, method="isotonic", cv="prefit")
    calibrator_isotonic.fit(val_stack, y_val)
    val_prob_iso = calibrator_isotonic.predict_proba(val_stack)[:, 1]
    metrics_val_iso = evaluate_metrics(y_val.to_numpy(), val_prob_iso)

    print(f"  Validation RAW Meta:  AUC={metrics_val_raw['roc_auc']:.4f}, Brier={metrics_val_raw['brier_score']:.4f}, ECE={metrics_val_raw['ece']:.4f}")
    print(f"  Validation SIGMOID:   AUC={metrics_val_sig['roc_auc']:.4f}, Brier={metrics_val_sig['brier_score']:.4f}, ECE={metrics_val_sig['ece']:.4f}")
    print(f"  Validation ISOTONIC:  AUC={metrics_val_iso['roc_auc']:.4f}, Brier={metrics_val_iso['brier_score']:.4f}, ECE={metrics_val_iso['ece']:.4f}")

    if metrics_val_sig["ece"] <= metrics_val_iso["ece"]:
        selected_calibrator = calibrator_sigmoid
        calibrator_name = "Sigmoid (Platt Scaling)"
        metrics_val_selected = metrics_val_sig
    else:
        selected_calibrator = calibrator_isotonic
        calibrator_name = "Isotonic Regression"
        metrics_val_selected = metrics_val_iso
    print(f"  ==> Selected Calibrator: {calibrator_name}")

    # 8. Final Evaluation on Untouched Test Set (N=4,000)
    print("\n[STEP 8] Final Evaluation on Untouched Test Set (N=4,000)...")
    X_test_lstm = X_test_proc.reshape((-1, timesteps, 1))
    test_bilstm_p = bilstm_final.predict(X_test_lstm, verbose=0).ravel()
    test_svm_p = svm_final.predict_proba(X_test_proc)[:, 1]
    test_rf_p = rf_final.predict_proba(X_test_proc)[:, 1]
    test_log_p = logreg_final.predict_proba(X_test_proc)[:, 1]
    test_stack = np.column_stack([test_bilstm_p, test_svm_p, test_rf_p, test_log_p])

    test_raw_p = meta_model.predict_proba(test_stack)[:, 1]
    test_cal_p = selected_calibrator.predict_proba(test_stack)[:, 1]

    metrics_test_bilstm = evaluate_metrics(y_test.to_numpy(), test_bilstm_p)
    metrics_test_svm = evaluate_metrics(y_test.to_numpy(), test_svm_p)
    metrics_test_rf = evaluate_metrics(y_test.to_numpy(), test_rf_p)
    metrics_test_log = evaluate_metrics(y_test.to_numpy(), test_log_p)
    metrics_test_raw = evaluate_metrics(y_test.to_numpy(), test_raw_p)
    metrics_test_cal = evaluate_metrics(y_test.to_numpy(), test_cal_p)
    ci_test_cal = bootstrap_ci(y_test.to_numpy(), test_cal_p)

    print("\n" + "=" * 80)
    print("FINAL TEST SET PERFORMANCE SUMMARY (N=4,000 held-out clinical samples)")
    print("=" * 80)
    models_summary = [
        ("Bi-LSTM", metrics_test_bilstm),
        ("SVM (RBF)", metrics_test_svm),
        ("Random Forest", metrics_test_rf),
        ("Logistic Regression", metrics_test_log),
        ("Raw Meta-Stacking", metrics_test_raw),
        (f"Calibrated Ensemble ({calibrator_name})", metrics_test_cal)
    ]
    for name, m in models_summary:
        print(f"{name:<35} | Acc: {m['accuracy']:.4f} | Sens: {m['sensitivity']:.4f} | Spec: {m['specificity']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | Brier: {m['brier_score']:.4f} | ECE: {m['ece']:.4f}")

    print(f"\nCalibrated Ensemble 95% CIs:")
    print(f"  ROC-AUC 95% CI: [{ci_test_cal['roc_auc_ci95'][0]:.4f}, {ci_test_cal['roc_auc_ci95'][1]:.4f}]")
    print(f"  Brier Score 95% CI: [{ci_test_cal['brier_score_ci95'][0]:.4f}, {ci_test_cal['brier_score_ci95'][1]:.4f}]")
    print(f"  ECE 95% CI: [{ci_test_cal['ece_ci95'][0]:.4f}, {ci_test_cal['ece_ci95'][1]:.4f}]")

    # 9. Diagnostic Visualizations
    print("\n[STEP 9] Generating Diagnostic Visualizations...")
    
    # A. ROC Comparison Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba, color, ls in [
        ("Bi-LSTM", test_bilstm_p, "#8b5cf6", "--"),
        ("SVM", test_svm_p, "#3b82f6", "--"),
        ("Random Forest", test_rf_p, "#10b981", "--"),
        ("LogReg", test_log_p, "#f59e0b", "--"),
        ("Calibrated Quad-Ensemble", test_cal_p, "#ef4444", "-")
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        score = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})", color=color, linestyle=ls, lw=2.2 if "-" == ls else 1.5)
    ax.plot([0, 1], [0, 1], "k:", label="Random Guess (AUC = 0.5000)")
    ax.set_title("ROC Curves — Quad-Model Stacking Ensemble", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "roc_comparison.png"), dpi=200)
    plt.close(fig)

    # B. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba, color, ls in [
        ("Bi-LSTM", test_bilstm_p, "#8b5cf6", "--"),
        ("SVM", test_svm_p, "#3b82f6", "--"),
        ("Random Forest", test_rf_p, "#10b981", "--"),
        ("LogReg", test_log_p, "#f59e0b", "--"),
        ("Calibrated Quad-Ensemble", test_cal_p, "#ef4444", "-")
    ]:
        p, r, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(r, p, label=f"{name} (PR-AUC = {ap:.4f})", color=color, linestyle=ls, lw=2.2 if "-" == ls else 1.5)
    ax.set_title("Precision-Recall Curves — Quad-Model Stacking Ensemble", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pr_comparison.png"), dpi=200)
    plt.close(fig)

    # C. Calibration Curve (Reliability Diagram)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba, color in [
        ("Raw Stacking", test_raw_p, "#f59e0b"),
        ("Calibrated Ensemble", test_cal_p, "#ef4444")
    ]:
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", label=f"{name} (ECE = {compute_ece(y_test, proba):.4f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax.set_title("Probability Calibration Reliability Diagram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of True Wilson Disease Positives")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "calibration_comparison.png"), dpi=200)
    plt.close(fig)

    # D. Confusion Matrix
    y_pred_cal = (test_cal_p >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_cal)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix — Calibrated Quad-Ensemble", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis (0=Negative, 1=Wilson Disease)", fontweight="bold")
    ax.set_ylabel("True Diagnosis", fontweight="bold")
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # E. Feature Importance from Random Forest
    try:
        proc_feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        proc_feature_names = [f"F_{i}" for i in range(timesteps)]
    
    importances = rf_final.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(15), importances[sorted_idx][::-1], color="#10b981", align="center")
    clean_names = [proc_feature_names[i].replace("num__", "").replace("cat__", "") for i in sorted_idx][::-1]
    ax.set_yticks(range(15))
    ax.set_yticklabels(clean_names, fontsize=9)
    ax.set_xlabel("Gini Feature Importance")
    ax.set_title("Top 15 Predictive Clinical Biomarkers (Random Forest)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=200)
    plt.close(fig)

    # 10. Serialization of All v3.0.0 Artifacts
    print("\n[STEP 10] Serializing Enhanced Artifacts to wilson_artifacts/...")
    joblib.dump(preprocessor, os.path.join(OUT_DIR, "preprocessor.joblib"))
    joblib.dump(svm_final, os.path.join(OUT_DIR, "svm.joblib"))
    joblib.dump(rf_final, os.path.join(OUT_DIR, "rf.joblib"))
    joblib.dump(logreg_final, os.path.join(OUT_DIR, "logreg_base.joblib"))
    joblib.dump(meta_model, os.path.join(OUT_DIR, "meta_model.joblib"))
    joblib.dump(selected_calibrator, os.path.join(OUT_DIR, "calibrator.joblib"))

    bilstm_keras_path = os.path.join(OUT_DIR, "bilstm_model.keras")
    bilstm_final.save(bilstm_keras_path)
    print(f"  - Saved Bi-LSTM model to {bilstm_keras_path}")

    # Generate background distribution for SHAP
    np.random.seed(RANDOM_STATE)
    shap_bg = X_train_proc[np.random.choice(len(X_train_proc), size=min(100, len(X_train_proc)), replace=False)]
    np.save(os.path.join(OUT_DIR, "shap_background.npy"), shap_bg)
    joblib.dump(shap_bg, os.path.join(OUT_DIR, "shap_background.joblib"))
    print("  - Saved SHAP background samples")

    meta_info = {
        "model_version": MODEL_VERSION,
        "constituent_models": ["Bi-LSTM", "SVM", "RandomForest", "LogisticRegression"],
        "meta_model": "LogisticRegression (4-Model Stacking)",
        "calibration_method": calibrator_name,
        "validation_metrics": metrics_val_selected,
        "test_metrics_calibrated": metrics_test_cal,
        "test_metrics_raw": metrics_test_raw,
        "test_metrics_bilstm": metrics_test_bilstm,
        "test_metrics_svm": metrics_test_svm,
        "test_metrics_rf": metrics_test_rf,
        "test_metrics_logreg": metrics_test_log,
        "test_ci95": ci_test_cal,
        "features": list(X_train_raw.columns),
        "processed_features": proc_feature_names,
        "timestamp": pd.Timestamp.utcnow().isoformat()
    }
    with open(os.path.join(OUT_DIR, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] QUAD-MODEL STACKING ENSEMBLE ({MODEL_VERSION}) TRAINED & SERIALIZED!")
    print("=" * 80)


if __name__ == "__main__":
    run_training_pipeline()
