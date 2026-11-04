#!/usr/bin/env python3
"""
scripts/train_v2_remediation_pipeline.py

Phase 8 Increment 2: Comprehensive Multi-Model Remediation Pipeline
1. Leakage-free preprocessing (Split first -> Fit preprocessor on train only -> Transform val & test).
2. Deep Recurrent Bi-LSTM + Non-linear SVM + Linear Logistic Regression base models.
3. 5-Fold Stratified Out-of-fold (OOF) base model predictions on training split for meta-classifier training.
4. Tri-model Stacking Meta-Classifier (Bi-LSTM, SVM, LogReg -> Meta Logistic Regression).
5. Probability calibration (Sigmoid vs Isotonic vs Raw) tuned on validation set.
6. Final evaluation on held-out test set with bootstrap confidence intervals and ECE.
7. Complete serialization of all v2 artifacts (v2.0.0-calibrated), curves, and metadata.
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
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss, log_loss,
    precision_score, recall_score
)

DATA_PATH = "Wilson_disease_dataset_v2.csv"
OUT_DIR = "wilson_artifacts"
RANDOM_STATE = 42
TARGET_COL = "Is_Wilson_Disease"

# Set deterministic seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def build_bilstm(input_timesteps: int) -> tf.keras.Model:
    """
    Constructs a calibrated Bidirectional LSTM architecture for clinical feature sequence modeling.
    Input shape: (input_timesteps, 1) representing standardized clinical features.
    """
    model = Sequential([
        Input(shape=(input_timesteps, 1)),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
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
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "log_loss": float(ll),
        "ece": float(ece),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def bootstrap_ci(y_true, y_prob, n_bootstraps=500, seed=42):
    """Computes 95% Confidence Intervals via non-parametric bootstrapping."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_list, brier_list, ece_list = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
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

def run_pipeline():
    print("=" * 70)
    print("STARTING PHASE 8 INCREMENT 2 REMEDIATION PIPELINE (WITH BI-LSTM)")
    print("=" * 70)
    
    # 1. Load dataset
    print(f"\n[STEP 1] Loading synthetic dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    if "Name" in df.columns:
        df = df.drop(columns=["Name"])
    print(f"Loaded dataset shape: {df.shape}")

    X_raw = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL].astype(int)

    # 2. Strict Pre-Split Partitioning (Leakage Prevention)
    print("\n[STEP 2] Partitioning raw dataset into Train (60%), Val (20%), and Test (20%)...")
    if len(df) > 20000:
        print(f"Subsampling {len(df)} down to 20,000 stratified samples for balanced training...")
        X_sub, _, y_sub, _ = train_test_split(X_raw, y_raw, train_size=20000, stratify=y_raw, random_state=RANDOM_STATE)
    else:
        X_sub, y_sub = X_raw, y_raw

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.20, stratify=y_sub, random_state=RANDOM_STATE
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
    )
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"Partition sizes: Train={len(X_train_raw)}, Validation={len(X_val_raw)}, Test={len(X_test_raw)}")

    # 3. Fit Preprocessing exclusively on X_train_raw
    print("\n[STEP 3] Fitting Preprocessor on X_train_raw ONLY (zero test leakage)...")
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
    print(f"Preprocessed feature dimension: {timesteps} features.")

    # 4. Out-of-Fold (OOF) Stacking Pipeline on Train Split
    print("\n[STEP 4] Generating 5-Fold Stratified Out-of-Fold Predictions for 3 Base Models (Bi-LSTM, SVM, LogReg)...")
    n_train = len(X_train_proc)
    oof_bilstm = np.zeros(n_train)
    oof_svm = np.zeros(n_train)
    oof_log = np.zeros(n_train)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train)):
        print(f"  --> Training Fold {fold + 1}/5 (train={len(tr_idx)}, val={len(val_idx)})...")
        X_tr, y_tr = X_train_proc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train_proc[val_idx], y_train.iloc[val_idx]

        # A. Bi-LSTM Fold
        X_tr_lstm = X_tr.reshape((-1, timesteps, 1))
        X_va_lstm = X_va.reshape((-1, timesteps, 1))
        m_bilstm = build_bilstm(timesteps)
        es_fold = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        m_bilstm.fit(
            X_tr_lstm, y_tr,
            validation_data=(X_va_lstm, y_va),
            epochs=15, batch_size=64,
            callbacks=[es_fold], verbose=0
        )
        oof_bilstm[val_idx] = m_bilstm.predict(X_va_lstm, verbose=0).ravel()

        # B. SVM Fold
        m_svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
        m_svm.fit(X_tr, y_tr)
        oof_svm[val_idx] = m_svm.predict_proba(X_va)[:, 1]

        # C. LogReg Fold
        m_log = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
        m_log.fit(X_tr, y_tr)
        oof_log[val_idx] = m_log.predict_proba(X_va)[:, 1]

    # Stacking matrix ordered exactly as: [Bi-LSTM, SVM, LogReg]
    oof_stack_train = np.column_stack([oof_bilstm, oof_svm, oof_log])
    print(f"OOF Stack Matrix shape: {oof_stack_train.shape} (3 constituent model predictions)")

    # 5. Train Meta-Classifier exclusively on OOF Predictions
    print("\n[STEP 5] Training Stacking Meta-Classifier exclusively on OOF predictions...")
    meta_model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    meta_model.fit(oof_stack_train, y_train)
    print(f"Meta-Classifier weights [Bi-LSTM, SVM, LogReg]: {meta_model.coef_[0]}")
    print(f"Meta-Classifier Intercept: {meta_model.intercept_[0]:.4f}")

    # 6. Retrain Base Models on Full X_train_proc
    print("\n[STEP 6] Retraining all base models on full training split (N=12,000)...")
    
    # Bi-LSTM Final
    print("  Fitting final Bi-LSTM model...")
    bilstm_final = build_bilstm(timesteps)
    es_final = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    X_train_lstm = X_train_proc.reshape((-1, timesteps, 1))
    X_val_lstm = X_val_proc.reshape((-1, timesteps, 1))
    bilstm_history = bilstm_final.fit(
        X_train_lstm, y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=30, batch_size=64,
        callbacks=[es_final], verbose=1
    )

    # SVM Final
    print("  Fitting final SVM model...")
    svm_final = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    svm_final.fit(X_train_proc, y_train)

    # LogReg Final
    print("  Fitting final Logistic Regression model...")
    logreg_final = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    logreg_final.fit(X_train_proc, y_train)

    # 7. Generate Validation Stack Predictions
    print("\n[STEP 7] Evaluating Validation Split for Model Selection and Calibration...")
    val_bilstm_prob = bilstm_final.predict(X_val_lstm, verbose=0).ravel()
    val_svm_prob = svm_final.predict_proba(X_val_proc)[:, 1]
    val_log_prob = logreg_final.predict_proba(X_val_proc)[:, 1]
    val_stack = np.column_stack([val_bilstm_prob, val_svm_prob, val_log_prob])

    val_meta_raw_prob = meta_model.predict_proba(val_stack)[:, 1]

    # Evaluate Raw Meta on Validation
    metrics_val_raw = evaluate_metrics(y_val.to_numpy(), val_meta_raw_prob)
    print(f"Validation RAW Meta: AUC={metrics_val_raw['roc_auc']:.4f}, Brier={metrics_val_raw['brier_score']:.4f}, ECE={metrics_val_raw['ece']:.4f}")

    # 8. Fit Probability Calibrators on Validation Set
    print("\n[STEP 8] Fitting Sigmoid (Platt) and Isotonic Calibrators on Validation Set...")
    calibrator_sigmoid = CalibratedClassifierCV(estimator=meta_model, method="sigmoid", cv="prefit")
    calibrator_sigmoid.fit(val_stack, y_val)
    val_prob_sig = calibrator_sigmoid.predict_proba(val_stack)[:, 1]
    metrics_val_sig = evaluate_metrics(y_val.to_numpy(), val_prob_sig)

    calibrator_isotonic = CalibratedClassifierCV(estimator=meta_model, method="isotonic", cv="prefit")
    calibrator_isotonic.fit(val_stack, y_val)
    val_prob_iso = calibrator_isotonic.predict_proba(val_stack)[:, 1]
    metrics_val_iso = evaluate_metrics(y_val.to_numpy(), val_prob_iso)

    print(f"Validation SIGMOID: AUC={metrics_val_sig['roc_auc']:.4f}, Brier={metrics_val_sig['brier_score']:.4f}, ECE={metrics_val_sig['ece']:.4f}")
    print(f"Validation ISOTONIC: AUC={metrics_val_iso['roc_auc']:.4f}, Brier={metrics_val_iso['brier_score']:.4f}, ECE={metrics_val_iso['ece']:.4f}")

    # Selection rule: Pick calibrator minimizing Brier score and ECE on validation set
    if metrics_val_sig["ece"] <= metrics_val_iso["ece"]:
        selected_calibrator = calibrator_sigmoid
        calibrator_name = "Sigmoid (Platt Scaling)"
        metrics_val_selected = metrics_val_sig
    else:
        selected_calibrator = calibrator_isotonic
        calibrator_name = "Isotonic Regression"
        metrics_val_selected = metrics_val_iso
    print(f"\n--> SELECTED CALIBRATOR: {calibrator_name}")

    # 9. Final Verification on Untouched Test Set
    print("\n[STEP 9] Evaluating FINAL Models on UNTOUCHED Test Set (N=4,000)...")
    X_test_lstm = X_test_proc.reshape((-1, timesteps, 1))
    test_bilstm_prob = bilstm_final.predict(X_test_lstm, verbose=0).ravel()
    test_svm_prob = svm_final.predict_proba(X_test_proc)[:, 1]
    test_log_prob = logreg_final.predict_proba(X_test_proc)[:, 1]
    test_stack = np.column_stack([test_bilstm_prob, test_svm_prob, test_log_prob])

    test_raw_prob = meta_model.predict_proba(test_stack)[:, 1]
    test_cal_prob = selected_calibrator.predict_proba(test_stack)[:, 1]

    metrics_test_bilstm = evaluate_metrics(y_test.to_numpy(), test_bilstm_prob)
    metrics_test_svm = evaluate_metrics(y_test.to_numpy(), test_svm_prob)
    metrics_test_log = evaluate_metrics(y_test.to_numpy(), test_log_prob)
    metrics_test_raw = evaluate_metrics(y_test.to_numpy(), test_raw_prob)
    metrics_test_cal = evaluate_metrics(y_test.to_numpy(), test_cal_prob)
    ci_test_cal = bootstrap_ci(y_test.to_numpy(), test_cal_prob)

    print("\n" + "=" * 80)
    print("FINAL TEST SET PERFORMANCE COMPARISON (ALL CONSTITUENTS & ENSEMBLE):")
    print("=" * 80)
    print(f"{'Model':<22} | {'ROC-AUC':<10} | {'Brier':<10} | {'ECE':<10} | {'Accuracy':<10} | {'Recall':<10}")
    print("-" * 80)
    print(f"{'Bi-LSTM':<22} | {metrics_test_bilstm['roc_auc']:<10.4f} | {metrics_test_bilstm['brier_score']:<10.4f} | {metrics_test_bilstm['ece']:<10.4f} | {metrics_test_bilstm['accuracy']:<10.4f} | {metrics_test_bilstm['sensitivity']:<10.4f}")
    print(f"{'SVM (RBF)':<22} | {metrics_test_svm['roc_auc']:<10.4f} | {metrics_test_svm['brier_score']:<10.4f} | {metrics_test_svm['ece']:<10.4f} | {metrics_test_svm['accuracy']:<10.4f} | {metrics_test_svm['sensitivity']:<10.4f}")
    print(f"{'Logistic Regression':<22} | {metrics_test_log['roc_auc']:<10.4f} | {metrics_test_log['brier_score']:<10.4f} | {metrics_test_log['ece']:<10.4f} | {metrics_test_log['accuracy']:<10.4f} | {metrics_test_log['sensitivity']:<10.4f}")
    print(f"{'Raw Stacking Meta':<22} | {metrics_test_raw['roc_auc']:<10.4f} | {metrics_test_raw['brier_score']:<10.4f} | {metrics_test_raw['ece']:<10.4f} | {metrics_test_raw['accuracy']:<10.4f} | {metrics_test_raw['sensitivity']:<10.4f}")
    print(f"{'Calibrated Ensemble':<22} | {metrics_test_cal['roc_auc']:<10.4f} | {metrics_test_cal['brier_score']:<10.4f} | {metrics_test_cal['ece']:<10.4f} | {metrics_test_cal['accuracy']:<10.4f} | {metrics_test_cal['sensitivity']:<10.4f}")
    print("-" * 80)
    print(f"Calibrated 95% CIs -> AUC: [{ci_test_cal['roc_auc_ci95'][0]:.4f}, {ci_test_cal['roc_auc_ci95'][1]:.4f}], ECE: [{ci_test_cal['ece_ci95'][0]:.4f}, {ci_test_cal['ece_ci95'][1]:.4f}]")

    pct_intermediate_cal = np.mean((test_cal_prob >= 0.20) & (test_cal_prob <= 0.80)) * 100

    # 10. Generate Visual Artifacts
    print("\n[STEP 10] Generating Evaluation Curves & Visualizations...")
    os.makedirs(OUT_DIR, exist_ok=True)

    # A. Bi-LSTM History Plots
    if bilstm_history:
        h = bilstm_history.history
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h["loss"], label="Train Loss", color="#4f46e5", lw=2)
        if "val_loss" in h:
            ax.plot(h["val_loss"], label="Val Loss", color="#06b6d4", lw=2)
        ax.set_title("Bi-LSTM Training & Validation Loss", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(OUT_DIR, "bilstm_history_loss.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        acc_key = "accuracy" if "accuracy" in h else "acc"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h[acc_key], label="Train Accuracy", color="#10b981", lw=2)
        val_acc_key = f"val_{acc_key}"
        if val_acc_key in h:
            ax.plot(h[val_acc_key], label="Val Accuracy", color="#f59e0b", lw=2)
        ax.set_title("Bi-LSTM Training & Validation Accuracy", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(OUT_DIR, "bilstm_history_acc.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # B. Reliability Curves (Calibration)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p, col, ls in [
        ("Bi-LSTM", test_bilstm_prob, "#8b5cf6", "--"),
        ("SVM", test_svm_prob, "#3b82f6", "--"),
        ("LogReg", test_log_prob, "#64748b", "--"),
        ("Raw Meta", test_raw_prob, "#ef4444", "--"),
        (f"Calibrated ({calibrator_name})", test_cal_prob, "#10b981", "-")
    ]:
        pt, pp = calibration_curve(y_test, p, n_bins=10)
        e = compute_ece(y_test.to_numpy(), p)
        ax.plot(pp, pt, marker='o' if ls=='-' else 's', linestyle=ls, label=f"{name} (ECE={e:.3f})", color=col, lw=2 if ls=='-' else 1.2)
    ax.plot([0, 1], [0, 1], 'k:', label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Observed Fraction of Positives", fontsize=11)
    ax.set_title("Reliability Diagram: Constituent Models vs Calibrated Ensemble", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "calibration_StackedMeta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual model calibration curves
    for name, proba in [("BiLSTM", test_bilstm_prob), ("SVM", test_svm_prob), ("LogReg", test_log_prob)]:
        fig, ax = plt.subplots(figsize=(6, 5))
        pt, pp = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(pp, pt, marker='o', label=name, color='#6366f1')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(f"Calibration Curve: {name}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
        fig.savefig(os.path.join(OUT_DIR, f"calibration_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # C. Multi-Model ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p, col, ls in [
        ("Bi-LSTM", test_bilstm_prob, "#8b5cf6", "--"),
        ("SVM", test_svm_prob, "#3b82f6", "--"),
        ("LogReg", test_log_prob, "#64748b", "--"),
        ("Calibrated Ensemble", test_cal_prob, "#10b981", "-")
    ]:
        fpr, tpr, _ = roc_curve(y_test, p)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})", color=col, linestyle=ls, lw=2 if ls=='-' else 1.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison — Tri-Model Ensemble & Constituents", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "roc_StackedMeta.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "roc_all_models.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual ROC curves
    for name, proba in [("BiLSTM", test_bilstm_prob), ("SVM", test_svm_prob), ("LogReg", test_log_prob)]:
        fig, ax = plt.subplots(figsize=(6, 5))
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}", color='#4f46e5', lw=2)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f"ROC Curve: {name}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(OUT_DIR, f"roc_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # D. Multi-Model PR Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p, col, ls in [
        ("Bi-LSTM", test_bilstm_prob, "#8b5cf6", "--"),
        ("SVM", test_svm_prob, "#3b82f6", "--"),
        ("LogReg", test_log_prob, "#64748b", "--"),
        ("Calibrated Ensemble", test_cal_prob, "#10b981", "-")
    ]:
        prec, rec, _ = precision_recall_curve(y_test, p)
        ap = average_precision_score(y_test, p)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=col, linestyle=ls, lw=2 if ls=='-' else 1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Comparison — Tri-Model Ensemble & Constituents", fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "pr_StackedMeta.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "pr_all_models.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual PR curves
    for name, proba in [("BiLSTM", test_bilstm_prob), ("SVM", test_svm_prob), ("LogReg", test_log_prob)]:
        fig, ax = plt.subplots(figsize=(6, 5))
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=f"AP={average_precision_score(y_test, proba):.3f}", color='#06b6d4', lw=2)
        ax.set_title(f"Precision-Recall Curve: {name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        fig.savefig(os.path.join(OUT_DIR, f"pr_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # E. Confusion Matrices
    for name, proba in [
        ("StackedMeta", test_cal_prob),
        ("BiLSTM", test_bilstm_prob),
        ("SVM", test_svm_prob),
        ("LogReg", test_log_prob)
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, (proba >= 0.5).astype(int))
        im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.colorbar(im, ax=ax)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.savefig(os.path.join(OUT_DIR, f"confusion_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 11. SHAP Background Sampling
    print("\n[STEP 11] Creating SHAP Background Sample from preprocessed training data...")
    bg_samples = X_train_proc[:50]
    joblib.dump(bg_samples, os.path.join(OUT_DIR, "shap_background.joblib"))

    # 12. Save All Model Artifacts
    print("\n[STEP 12] Serializing v2 Model Artifacts...")
    joblib.dump(preprocessor, os.path.join(OUT_DIR, "preprocessor.joblib"))
    joblib.dump(svm_final, os.path.join(OUT_DIR, "svm.joblib"))
    joblib.dump(logreg_final, os.path.join(OUT_DIR, "logreg_base.joblib"))
    joblib.dump(meta_model, os.path.join(OUT_DIR, "meta_model.joblib"))
    joblib.dump(selected_calibrator, os.path.join(OUT_DIR, "calibrator.joblib"))

    # Save Bi-LSTM in both .keras and .h5 formats for maximum interoperability
    bilstm_keras_path = os.path.join(OUT_DIR, "bilstm_model.keras")
    bilstm_h5_path = os.path.join(OUT_DIR, "bilstm_model.h5")
    try:
        bilstm_final.save(bilstm_keras_path)
        print(f"Saved Bi-LSTM to {bilstm_keras_path}")
    except Exception as e:
        print("Keras save warning:", e)
    try:
        bilstm_final.save(bilstm_h5_path)
        print(f"Saved Bi-LSTM to {bilstm_h5_path}")
    except Exception as e:
        print("H5 save warning:", e)

    # Save classification reports for all models
    for name, proba in [
        ("StackedMeta", test_cal_prob),
        ("BiLSTM", test_bilstm_prob),
        ("SVM", test_svm_prob),
        ("LogReg", test_log_prob)
    ]:
        rep = classification_report(y_test, (proba >= 0.5).astype(int), output_dict=True)
        with open(os.path.join(OUT_DIR, f"classif_report_{name}.json"), "w") as f:
            json.dump(rep, f, indent=2)

    # Save processed feature names
    try:
        proc_feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        proc_feature_names = None

    meta_info = {
        "model_version": "v2.0.0-calibrated",
        "constituent_models": ["Bi-LSTM", "SVM", "LogisticRegression"],
        "meta_model": "LogisticRegression (Stacking)",
        "calibration_method": calibrator_name,
        "validation_metrics": metrics_val_selected,
        "test_metrics_calibrated": metrics_test_cal,
        "test_metrics_raw": metrics_test_raw,
        "test_metrics_bilstm": metrics_test_bilstm,
        "test_metrics_svm": metrics_test_svm,
        "test_metrics_logreg": metrics_test_log,
        "test_ci95": ci_test_cal,
        "features": list(X_train_raw.columns),
        "processed_features": proc_feature_names,
        "intermediate_prob_percentage": float(pct_intermediate_cal),
        "timestamp": pd.Timestamp.utcnow().isoformat()
    }
    with open(os.path.join(OUT_DIR, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print("\n" + "=" * 70)
    print("[SUCCESS] TRI-MODEL REMEDIATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved to {OUT_DIR}/:")
    print("  - preprocessor.joblib")
    print("  - bilstm_model.keras & bilstm_model.h5")
    print("  - svm.joblib")
    print("  - logreg_base.joblib")
    print("  - meta_model.joblib (3-model stacking)")
    print(f"  - calibrator.joblib ({calibrator_name})")
    print("  - shap_background.joblib")
    print("  - meta_info.json")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
