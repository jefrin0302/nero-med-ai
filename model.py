#!/usr/bin/env python3
"""
model.py

Clinical 4-Model Stacking Ensemble (XGBoost + Logistic Regression + Random Forest + SVM)
with Median Imputation + RobustScaler (IQR) + ColumnTransformer Preprocessing,
Out-Of-Fold (OOF) Stacking Meta-Classifier, Isotonic Probability Calibration,
and comprehensive clinical metrics & evaluation artifact generation.

Dataset: Wilson_disease_dataset.csv
Outputs: ./wilson_artifacts/
"""

import os
import json
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss, log_loss
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    raise ImportError("xgboost is required for this pipeline. Please install via: pip install xgboost")

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = "Wilson_disease_dataset.csv"
OUT_DIR = "wilson_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
TARGET_COL = "Is_Wilson_Disease"

# ----------------------------
# Utility plotting functions
# ----------------------------
def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def compute_ece(y_true, probas, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i + 1]
        mask = (probas >= low) & (probas <= high if i == n_bins - 1 else probas < high)
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probas[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
    return float(ece)

def calculate_full_metrics(y_true, proba, pred):
    """Calculates full clinical evaluation metrics."""
    acc = float(accuracy_score(y_true, pred))
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    
    try:
        roc_auc = float(roc_auc_score(y_true, proba))
    except Exception:
        roc_auc = 0.0
        
    try:
        pr_auc = float(average_precision_score(y_true, proba))
    except Exception:
        pr_auc = 0.0
        
    brier = float(brier_score_loss(y_true, proba))
    try:
        lloss = float(log_loss(y_true, proba))
    except Exception:
        lloss = 0.0
        
    ece = compute_ece(y_true, proba, n_bins=10)
    
    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": lloss,
        "ece": ece
    }

def plot_confusion_matrix(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    savefig(fig, fname)

# ----------------------------
# Data Loading & Leakage Pruning
# ----------------------------
def load_df(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Please place 'Wilson_disease_dataset.csv' here.")
    df = pd.read_csv(path)
    # Ensure Alcohol Use is consistently treated as string for OneHotEncoder
    if "Alcohol Use" in df.columns:
        df["Alcohol Use"] = df["Alcohol Use"].astype(str)
    print(f"Loaded dataset: {df.shape}")
    return df

def detect_and_remove_leakage(df, target_col=TARGET_COL, verbose=True):
    """
    Checks for artificial synthetic leakage columns (e.g. ID, patient identifiers).
    Preserves core clinical biomarkers and demographic variables.
    """
    removed = []
    # Core 23 clinical features must be preserved
    protected_features = {
        "Age", "Sex", "Ceruloplasmin Level", "Copper in Blood Serum",
        "Free Copper in Blood Serum", "Copper in Urine", "ALT", "AST",
        "Total Bilirubin", "Albumin", "Alkaline Phosphatase (ALP)",
        "Prothrombin Time / INR", "Gamma-Glutamyl Transferase (GGT)",
        "Kayser-Fleischer Rings", "Neurological Symptoms Score",
        "Psychiatric Symptoms", "Cognitive Function Score",
        "Family History", "ATB7B Gene Mutation", "Region",
        "Socioeconomic Status", "Alcohol Use", "BMI"
    }
    cols = [c for c in df.columns if c != target_col and c not in protected_features]
    y = df[target_col]
    for col in cols:
        ser = df[col]
        try:
            mapping = df.groupby(col)[target_col].agg(lambda x: x.mode().iat[0] if len(x.mode()) > 0 else x.iloc[0])
            pred = ser.map(mapping)
            acc = (pred == y).mean()
        except Exception:
            acc = 0.0
        if acc == 1.0:
            removed.append(col)
            if verbose:
                print(f"[LEAKAGE] Column '{col}' dropped.")
    if removed:
        df = df.drop(columns=removed)
    return df, removed

# ----------------------------
# Robust Preprocessing Builder
# ----------------------------
def build_preprocessor_and_features(df, target_col=TARGET_COL):
    """
    Constructs the clinical preprocessing pipeline:
    - 19 Numeric features: Median Imputation + RobustScaler(IQR)
    - 4 Categorical features: Most Frequent Imputation + OneHotEncoder
    Total: 23 clinical features -> 30 processed features
    """
    X = df.drop(columns=[target_col])
    if "Name" in X.columns:
        X = X.drop(columns=["Name"])
    y = df[target_col].astype(int)

    categorical_features = ["Sex", "Region", "Socioeconomic Status", "Alcohol Use"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # Optimal Robust Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(quantile_range=(25.0, 75.0)))  # Immune to extreme liver enzyme spikes
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    X_proc = preprocessor.fit_transform(X)

    # Reconstruct human-readable processed feature names
    cat_names = []
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_features).tolist()
    except Exception:
        cat_names = categorical_features
    processed_features = [f"num__{c}" for c in numeric_features] + [f"cat__{c}" for c in cat_names]

    return X_proc, y, preprocessor, numeric_features + categorical_features, processed_features

# ----------------------------
# Main Training & Evaluation Pipeline
# ----------------------------
def train_and_save():
    print("=" * 60)
    print("Starting 4-Model Stacking Training Pipeline (Wilson's Disease)")
    print("=" * 60)
    t_start = time.time()

    # 1. Load dataset
    df = load_df(DATA_PATH)
    if "Name" in df.columns:
        df = df.drop(columns=["Name"])
        print("Dropped non-predictive 'Name' column.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")

    # 2. Leakage check
    df_clean, removed_cols = detect_and_remove_leakage(df, TARGET_COL)
    if removed_cols:
        with open(os.path.join(OUT_DIR, "leakage_removed.json"), "w") as f:
            json.dump({"removed_columns": removed_cols}, f, indent=2)

    # 3. Build Robust Preprocessor
    X_proc, y, preprocessor, raw_features, processed_features = build_preprocessor_and_features(df_clean, TARGET_COL)

    # 4. Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Dataset split: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows (Features: {X_train.shape[1]})")

    # 5. Define the 4 Constituent Models
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

    # --------------------
    # Optional Bi-LSTM
    # --------------------
    bilstm_history = None
    if TF_AVAILABLE:
        timesteps = X_train.shape[1]
        X_train_lstm = X_train.reshape((X_train.shape[0], timesteps, 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], timesteps, 1))
        print("Training Bi-LSTM...")
        bilstm = build_bilstm(timesteps)
        es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        hist = bilstm.fit(X_train_lstm, y_train, validation_split=0.15, epochs=25, batch_size=32, callbacks=[es], verbose=1)
        bilstm_history = hist.history
        proba_bilstm = bilstm.predict(X_test_lstm).ravel()
        pred_bilstm = (proba_bilstm >= 0.5).astype(int)
        test_probas["BiLSTM"] = proba_bilstm
        test_preds["BiLSTM"] = pred_bilstm
        model_objects["BiLSTM"] = bilstm
        # save bilstm model
        bilstm_path = os.path.join(OUT_DIR, "bilstm_model.h5")
        try:
            bilstm.save(bilstm_path)
            print("Saved Bi-LSTM to", bilstm_path)
        except Exception as e:
            print("H5 save notice:", e)
        try:
            bilstm.save(os.path.join(OUT_DIR, "bilstm_model.keras"))
            print("Saved Bi-LSTM to bilstm_model.keras")
        except Exception as e:
            print("Keras save notice:", e)
        # Save history plots
        plot_history(bilstm_history, fname_prefix="bilstm_history")
    else:
        print("Skipping Bi-LSTM (TensorFlow not installed).")

    # 6. Stratified 5-Fold Cross Validation for Out-Of-Fold (OOF) Stacking
    print("\n--- 5-Fold Stratified Cross-Validation for Stacking Meta-Features ---")
    meta_train = np.zeros((X_train.shape[0], n_models))
    meta_test = np.zeros((X_test.shape[0], n_models))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Store out-of-fold predictions
    oof_predictions = {name: np.zeros(X_train.shape[0]) for name in model_names}
    fold_test_preds = {name: np.zeros((kf.get_n_splits(), X_test.shape[0])) for name in model_names}

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Processing Fold {fold_idx + 1}/5...")
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx] if isinstance(y_train, pd.Series) else y_train[tr_idx], \
                      y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

        for name in model_names:
            clf = models[name]
            # Clone model for fold
            from sklearn.base import clone
            fold_clf = clone(clf)
            fold_clf.fit(X_tr, y_tr)
            
            p_val = fold_clf.predict_proba(X_val)[:, 1]
            p_test = fold_clf.predict_proba(X_test)[:, 1]
            
            oof_predictions[name][val_idx] = p_val
            fold_test_preds[name][fold_idx, :] = p_test

    # Populate meta matrices [XGBoost, LogisticRegression, RandomForest, SVM]
    for i, name in enumerate(model_names):
        meta_train[:, i] = oof_predictions[name]
        meta_test[:, i] = fold_test_preds[name].mean(axis=0)

    # 7. Train Full Base Models on 100% of Training Data
    print("\n--- Fitting Full Base Models on 100% Training Data ---")
    fitted_base_models = {}
    test_base_probas = {}
    test_base_preds = {}

    for name in model_names:
        print(f"Training {name} on full training set...")
        clf = models[name]
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        
        fitted_base_models[name] = clf
        test_base_probas[name] = proba
        test_base_preds[name] = pred

    # 8. Train Stacking Meta-Classifier
    print("\n--- Training Stacking Meta-Classifier & Isotonic Calibrator ---")
    meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    meta_model.fit(meta_train, y_train)
    
    # Raw meta probability
    raw_meta_proba = meta_model.predict_proba(meta_test)[:, 1]
    raw_meta_pred = (raw_meta_proba >= 0.5).astype(int)

    # Post-hoc Isotonic Calibration
    calibrator = CalibratedClassifierCV(estimator=meta_model, method="isotonic", cv="prefit")
    calibrator.fit(meta_train, y_train)
    
    calibrated_meta_proba = calibrator.predict_proba(meta_test)[:, 1]
    calibrated_meta_pred = (calibrated_meta_proba >= 0.5).astype(int)

    # 9. Evaluate All Models on Holdout Test Set
    print("\n" + "=" * 60)
    print("HOLDOUT TEST EVALUATION RESULTS (N = %d)" % len(y_test))
    print("=" * 60)

    test_metrics = {}
    
    # Base models
    for name in model_names:
        metrics = calculate_full_metrics(y_test, test_base_probas[name], test_base_preds[name])
        test_metrics[name] = metrics
        print(f"{name:20s} | Acc: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f} | Recall: {metrics['recall']:.4f} | ECE: {metrics['ece']:.4f}")

    # Raw Meta
    raw_meta_metrics = calculate_full_metrics(y_test, raw_meta_proba, raw_meta_pred)
    test_metrics["StackedMeta_Raw"] = raw_meta_metrics
    print(f"{'StackedMeta (Raw)':20s} | Acc: {raw_meta_metrics['accuracy']:.4f} | AUC: {raw_meta_metrics['roc_auc']:.4f} | Recall: {raw_meta_metrics['recall']:.4f} | ECE: {raw_meta_metrics['ece']:.4f}")

    # Calibrated Meta (The Final Clinical Brain)
    calibrated_meta_metrics = calculate_full_metrics(y_test, calibrated_meta_proba, calibrated_meta_pred)
    test_metrics["StackedMeta_Calibrated"] = calibrated_meta_metrics
    print(f"{'StackedMeta (Calib)':20s} | Acc: {calibrated_meta_metrics['accuracy']:.4f} | AUC: {calibrated_meta_metrics['roc_auc']:.4f} | Recall: {calibrated_meta_metrics['recall']:.4f} | ECE: {calibrated_meta_metrics['ece']:.4f}")

    # 10. Generate Evaluation Plots
    print("\n--- Generating Visualization & Calibration Plots ---")
    
    # A. Combined ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in model_names:
        fpr, tpr, _ = roc_curve(y_test, test_base_probas[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})")
    # Calibrated Meta
    fpr, tpr, _ = roc_curve(y_test, calibrated_meta_proba)
    ax.plot(fpr, tpr, "k-", linewidth=2.2, label=f"Calibrated Meta-Ensemble (AUC={auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC Curves Comparison — 4-Model Clinical Ensemble", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.legend(loc="lower right", fontsize=9)
    savefig(fig, "roc_all_models.png")

    # B. Combined Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in model_names:
        prec, rec, _ = precision_recall_curve(y_test, test_base_probas[name])
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, test_base_probas[name]):.4f})")
    prec, rec, _ = precision_recall_curve(y_test, calibrated_meta_proba)
    ax.plot(rec, prec, "k-", linewidth=2.2, label=f"Calibrated Meta-Ensemble (AP={average_precision_score(y_test, calibrated_meta_proba):.4f})")
    ax.set_title("Precision-Recall Curves Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.legend(loc="lower left", fontsize=9)
    savefig(fig, "pr_all_models.png")

    # C. Calibration Reliability Diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in model_names:
        f_true, f_pred = calibration_curve(y_test, test_base_probas[name], n_bins=10)
        ax.plot(f_pred, f_true, marker="o", label=f"{name}")
    f_true, f_pred = calibration_curve(y_test, calibrated_meta_proba, n_bins=10)
    ax.plot(f_pred, f_true, "k-s", linewidth=2.5, label="Calibrated Meta (Isotonic)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    ax.set_title("Reliability Calibration Curves (Expected vs Predicted Risk)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction of Wilson Positives")
    ax.legend(loc="upper left", fontsize=9)
    savefig(fig, "calibration_comparison.png")

    # D. Confusion Matrices
    for name in model_names:
        plot_confusion_matrix(y_test, test_base_preds[name], f"Confusion Matrix — {name}", f"confusion_{name}.png")
    plot_confusion_matrix(y_test, calibrated_meta_pred, "Confusion Matrix — Calibrated Meta-Ensemble", "confusion_matrix.png")

    # E. Summary Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_names = model_names + ["Calibrated Meta"]
    accuracies = [test_metrics[n]["accuracy"] for n in model_names] + [calibrated_meta_metrics["accuracy"]]
    aucs = [test_metrics[n]["roc_auc"] for n in model_names] + [calibrated_meta_metrics["roc_auc"]]
    x = np.arange(len(plot_names))
    width = 0.35
    ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#3b82f6")
    ax.bar(x + width/2, aucs, width, label="ROC-AUC", color="#10b981")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_names, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0.98, 1.002)
    ax.set_ylabel("Metric Score")
    ax.set_title("Model Benchmark Performance Summary", fontsize=12, fontweight="bold")
    ax.legend()
    savefig(fig, "summary_metrics_comparison.png")

    # F. Multi-Model Feature Importance
    try:
        rf_imp = fitted_base_models["RandomForest"].feature_importances_
        xgb_imp = fitted_base_models["XGBoost"].feature_importances_
        avg_imp = (rf_imp + xgb_imp) / 2.0
        
        top_idx = np.argsort(avg_imp)[-15:]
        top_names = [processed_features[i].replace("num__", "").replace("cat__", "") for i in top_idx]
        top_scores = avg_imp[top_idx]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_names)), top_scores, color="#6366f1")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Consensus Tree Feature Importance (RF + XGBoost)")
        ax.set_title("Top 15 Predictive Clinical Biomarkers", fontsize=12, fontweight="bold")
        savefig(fig, "feature_importance.png")
    except Exception as e:
        print(f"Feature importance plot note: {e}")

    # 11. Serialize All Production Artifacts
    print("\n--- Serializing Models & Governance Artifacts to disk ---")
    joblib.dump(preprocessor, os.path.join(OUT_DIR, "preprocessor.joblib"))
    joblib.dump(fitted_base_models["XGBoost"], os.path.join(OUT_DIR, "xgb.joblib"))
    joblib.dump(fitted_base_models["LogisticRegression"], os.path.join(OUT_DIR, "logreg_base.joblib"))
    joblib.dump(fitted_base_models["RandomForest"], os.path.join(OUT_DIR, "rf.joblib"))
    joblib.dump(fitted_base_models["SVM"], os.path.join(OUT_DIR, "svm.joblib"))
    joblib.dump(meta_model, os.path.join(OUT_DIR, "meta_model.joblib"))
    joblib.dump(calibrator, os.path.join(OUT_DIR, "calibrator.joblib"))

    # Sample and serialize 100 background rows for SHAP KernelExplainer
    np.random.seed(RANDOM_STATE)
    bg_sample_idx = np.random.choice(X_train.shape[0], size=min(100, X_train.shape[0]), replace=False)
    shap_background = X_train[bg_sample_idx]
    joblib.dump(shap_background, os.path.join(OUT_DIR, "shap_background.joblib"))
    np.save(os.path.join(OUT_DIR, "shap_background.npy"), shap_background)

    # Serialize Comprehensive Meta Information & Governance Telemetry
    meta_info = {
        "model_version": "v4.0.0-robust-stacking",
        "constituent_models": [
            "XGBoost",
            "LogisticRegression",
            "RandomForest",
            "SVM"
        ],
        "meta_model": "LogisticRegression (4-Model Stacking)",
        "calibration_method": "Isotonic Regression",
        "preprocessing": "SimpleImputer(median) + RobustScaler(IQR 25-75) + OneHotEncoder",
        "validation_metrics": test_metrics["StackedMeta_Calibrated"],
        "test_metrics_calibrated": test_metrics["StackedMeta_Calibrated"],
        "test_metrics_raw": test_metrics["StackedMeta_Raw"],
        "test_metrics_xgb": test_metrics["XGBoost"],
        "test_metrics_logreg": test_metrics["LogisticRegression"],
        "test_metrics_rf": test_metrics["RandomForest"],
        "test_metrics_svm": test_metrics["SVM"],
        "features": raw_features,
        "processed_features": processed_features,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    with open(os.path.join(OUT_DIR, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTraining pipeline completed successfully in {total_time:.2f} seconds.")
    print(f"All 4 models + Meta + Calibrator + Preprocessor saved to: {os.path.abspath(OUT_DIR)}")
    return meta_info

if __name__ == "__main__":
    train_and_save()
