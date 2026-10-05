# app.py — Advanced SHAP UI dashboard + prediction endpoints (FINAL FIXED VERSION)
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import sys

# Prevent broken PyTorch DLL on Windows from crashing optional SHAP maskers
if 'torch' not in sys.modules:
    try:
        import torch
    except (OSError, ImportError):
        sys.modules['torch'] = None

import joblib
import numpy as np
import pandas as pd
import shap
import scipy
import sklearn
import scipy.sparse
import sklearn.compose
import sklearn.svm
import sklearn.preprocessing
import sklearn.impute

# Map numpy._core to numpy.core for numpy 1.x / 2.x cross-version pickle compatibility
if 'numpy._core' not in sys.modules:
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import json
import logging
from datetime import datetime

from rag_system import MedicalRAGSystem

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wilson_app")

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize RAG System
rag_assistant = MedicalRAGSystem()

ARTIFACT_DIR = "wilson_artifacts"

preprocessor = None
svm = None
logreg = None
meta = None
bilstm = None
shap_background = None
ensemble_explainer = None

def safe_load_artifact(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as err:
        logger.warning("Error loading artifact %s: %s", path, err)
        return None

try:
    preprocessor = safe_load_artifact(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))
    if preprocessor is not None and not hasattr(preprocessor, '_name_to_fitted_passthrough'):
        preprocessor._name_to_fitted_passthrough = {}
    svm = safe_load_artifact(os.path.join(ARTIFACT_DIR, "svm.joblib"))
    logreg = safe_load_artifact(os.path.join(ARTIFACT_DIR, "logreg_base.joblib"))
    meta = safe_load_artifact(os.path.join(ARTIFACT_DIR, "meta_model.joblib"))
    shap_background = safe_load_artifact(os.path.join(ARTIFACT_DIR, "shap_background.joblib"))

    # Load Bi-LSTM
    keras_path = os.path.join(ARTIFACT_DIR, "bilstm_model.keras")
    h5_path = os.path.join(ARTIFACT_DIR, "bilstm_model.h5")
    bilstm_target = keras_path if os.path.exists(keras_path) else (h5_path if os.path.exists(h5_path) else None)
    if bilstm_target:
        try:
            from tensorflow.keras.models import load_model
            bilstm = load_model(bilstm_target)
            logger.info("Bi-LSTM loaded successfully from %s", bilstm_target)
        except Exception as e:
            logger.warning("Bi-LSTM loading failed: %s", e)

    logger.info("Artifacts loaded: preproc=%s svm=%s logreg=%s meta=%s bilstm=%s shap_bg=%s",
                preprocessor is not None, svm is not None, logreg is not None, meta is not None, bilstm is not None, shap_background is not None)
except Exception as e:
    logger.exception("Artifact loading failed: %s", e)

def models_ready():
    return preprocessor is not None and svm is not None and logreg is not None and meta is not None

def full_ensemble_predict_proba(X_matrix):
    """
    Evaluates the complete multi-model ensemble pipeline:
    Bi-LSTM + SVM + Logistic Regression -> Stacking Meta-Classifier.
    Returns (N,) array of consensus Wilson Disease probabilities for input rows.
    """
    svm_p = svm.predict_proba(X_matrix)[:, 1]
    log_p = logreg.predict_proba(X_matrix)[:, 1]
    if getattr(meta, "n_features_in_", 2) == 3:
        if bilstm is not None:
            X_lstm = X_matrix.reshape((X_matrix.shape[0], X_matrix.shape[1], 1))
            bilstm_p = bilstm.predict(X_lstm, verbose=0).ravel()
        else:
            bilstm_p = (svm_p + log_p) / 2.0
        stacked = np.column_stack([bilstm_p, svm_p, log_p])
    else:
        stacked = np.column_stack([svm_p, log_p])
    return meta.predict_proba(stacked)[:, 1]

# Initialize Ensemble KernelExplainer at application startup
if shap_background is not None and models_ready():
    try:
        ensemble_explainer = shap.KernelExplainer(full_ensemble_predict_proba, shap_background)
        logger.info("Ensemble KernelExplainer initialized successfully.")
    except Exception as e:
        logger.warning("Failed to initialize Ensemble KernelExplainer: %s", e)


# ============================
# FIXED SAVE PLOT (NO BACKSLASHES)
# ============================
def save_plot(fig, name):
    try:
        path = os.path.join(UPLOAD_FOLDER, name)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved plot %s", path)

        # *** CRITICAL FIX ***  
        # Always return URL-ready POSIX path:
        return f"uploads/{name}".replace("\\", "/")

    except Exception as e:
        logger.exception("Failed to save plot %s: %s", name, e)
        try: plt.close(fig)
        except: pass
        return None


# ============================
# Feature name extraction
# ============================
def get_processed_feature_names(preproc):
    if preproc is None:
        return None
    try:
        raw_names = preproc.get_feature_names_out().tolist()
        clean = []
        for name in raw_names:
            if "__" in name:
                name = name.split("__", 1)[1]
            
            # Map encoded suffixes to readable medical terminology
            name = name.replace("Gamma-Glutamyl Transferase (GGT)", "GGT")
            name = name.replace("Kayser-Fleischer Rings_1", "KF Rings: Present")
            name = name.replace("Kayser-Fleischer Rings_0", "KF Rings: Absent")
            name = name.replace("ATB7B Gene Mutation_1", "ATP7B Mutation: Detected")
            name = name.replace("ATB7B Gene Mutation_0", "ATP7B Mutation: Normal")
            name = name.replace("Family History_1", "Family History: Positive")
            name = name.replace("Family History_0", "Family History: Negative")
            name = name.replace("Psychiatric Symptoms_1", "Psychiatric Symptoms: Yes")
            name = name.replace("Psychiatric Symptoms_0", "Psychiatric Symptoms: No")
            name = name.replace("Sex_Male", "Sex: Male")
            name = name.replace("Sex_Female", "Sex: Female")
            name = name.replace("Region_", "Region: ")
            name = name.replace("Socioeconomic Status_", "SES: ")
            clean.append(name)
        return clean
    except Exception as e:
        logger.warning("Feature name extraction failed: %s", e)
        return None


# ============================
# Generate randomized DF
# ============================
def generate_randomized_df(df_raw, n=20):
    rows = []
    for _ in range(n):
        r = {}
        for col in df_raw.columns:
            v = df_raw[col].iloc[0]
            if isinstance(v, (int, float, np.number)):
                if v == 0 or v is None:
                    r[col] = float(np.random.uniform(0.01, 1.0))
                else:
                    r[col] = float(np.random.uniform(max(0.01, v*0.3), v*1.8))
            else:
                r[col] = v
        rows.append(r)
    return pd.DataFrame(rows)


# ============================
# SHAP Helper
# ============================
def _shap_values_to_array(sv):
    if isinstance(sv, list):
        for v in sv:
            arr = np.array(v)
            if arr.ndim == 2:
                return arr
        return np.array(sv)
    return np.array(sv)


def create_shap_plots(explainer, shap_values, X, feature_names):
    out = {"waterfall": None}

    arr = _shap_values_to_array(shap_values)
    if arr is None: return out

    nfeat = arr.shape[1] if arr.ndim > 1 else len(arr)
    if feature_names is None or len(feature_names) != nfeat:
        feature_names = [f"Feature {i}" for i in range(nfeat)]

    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')

    # Waterfall plot (Patient Local Risk Attribution)
    try:
        plt.close('all')
        plt.figure(figsize=(10, 13))
        ev = float(np.array(explainer.expected_value).ravel()[0])
        first_row = arr[0] if arr.ndim > 1 else arr
        first_x = np.round(X[0], 2) if hasattr(X, '__getitem__') else None
        exp = shap.Explanation(
            values=first_row,
            base_values=ev,
            data=first_x,
            feature_names=feature_names
        )
        shap.plots.waterfall(exp, max_display=len(feature_names) + 1, show=False)
        fig = plt.gcf()
        out["waterfall"] = save_plot(fig, f"shap_waterfall_{timestamp}.png")
    except Exception as e:
        logger.warning("Waterfall plot failed: %s", e)

    return out


# ============================
# Safe SHAP + Prediction (Approach B: Full Ensemble KernelExplainer + Pure Patient Data)
# ============================
def safe_shap_and_predict(df_raw):
    bilstm_prob = None
    svm_prob = None
    log_prob = None
    final_prob = None

    # 1. Transform raw patient data (pure patient values)
    X = preprocessor.transform(df_raw)

    # 2. Existing multi-model ensemble prediction pipeline
    svm_p = float(svm.predict_proba(X)[:, 1][0])
    log_p = float(logreg.predict_proba(X)[:, 1][0])
    svm_prob = svm_p
    log_prob = log_p

    if getattr(meta, "n_features_in_", 2) == 3:
        if bilstm is not None:
            X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
            bilstm_p = float(bilstm.predict(X_lstm, verbose=0).ravel()[0])
            bilstm_prob = bilstm_p
        else:
            bilstm_p = (svm_p + log_p) / 2.0
            bilstm_prob = None
        stacked = np.column_stack([[bilstm_p], [svm_p], [log_p]])
    else:
        stacked = np.column_stack([[svm_p], [log_p]])

    final_prob = float(meta.predict_proba(stacked)[0][1])
    final_pred = int(final_prob >= 0.5)

    model_breakdown = {
        "bilstm": round(bilstm_prob * 100, 1) if bilstm_prob is not None else None,
        "svm": round(svm_prob * 100, 1) if svm_prob is not None else None,
        "logreg": round(log_prob * 100, 1) if log_prob is not None else None
    }

    # 3. Deterministic Full-Ensemble SHAP Explanation (Authentic patient data, zero noise)
    shap_results = {"waterfall": None}
    if ensemble_explainer is not None:
        try:
            import time
            t_start = time.time()
            np.random.seed(42)
            sv = ensemble_explainer.shap_values(X, nsamples=60, l1_reg=False, silent=True)
            t_elapsed = time.time() - t_start
            logger.info("Ensemble KernelExplainer generated SHAP values in %.2f seconds.", t_elapsed)

            # Validate array shape and features for SHAP 0.49.1 compatibility
            arr = _shap_values_to_array(sv)
            feature_names = get_processed_feature_names(preprocessor)
            shap_results = create_shap_plots(ensemble_explainer, arr, X, feature_names)
        except Exception as e:
            logger.warning("Ensemble SHAP explanation failed: %s", e)
    else:
        logger.warning("Ensemble KernelExplainer is not initialized; skipping SHAP plots.")

    # Return pure patient results: used_synthetic is False, df_raw is 100% authentic
    return final_prob, final_pred, shap_results, False, df_raw, model_breakdown


# ============================
# ROUTES
# ============================
@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/do")
def do():
    return render_template("do.html")


@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# ============================
# PWA ROUTES
# ============================
@app.route("/manifest.json")
def manifest():
    res = send_from_directory("static", "manifest.json")
    res.headers["Content-Type"] = "application/manifest+json"
    return res


@app.route("/service-worker.js")
def service_worker():
    res = send_from_directory("static", "service-worker.js")
    res.headers["Service-Worker-Allowed"] = "/"
    res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    res.headers["Content-Type"] = "application/javascript"
    return res


@app.route("/offline")
def offline():
    return render_template("offline.html")


VALID_REGIONS = {"East", "North", "South", "West"}


def parse_binary(val, field_name):
    if val is None or str(val).strip() == "":
        raise ValueError(f"Missing required field: {field_name}")
    s = str(val).strip()
    if s in ("1", "Yes", "yes", "True", "true"):
        return 1
    elif s in ("0", "No", "no", "False", "false"):
        return 0
    raise ValueError(f"Invalid value for {field_name}: '{val}'. Must be Yes/1 or No/0.")


def parse_region(val):
    if val is None or str(val).strip() == "":
        raise ValueError("Missing required field: Region")
    s = str(val).strip()
    match = {r.lower(): r for r in VALID_REGIONS}.get(s.lower())
    if match is None:
        raise ValueError(f"Invalid Region: '{val}'. Must be one of: East, North, South, West.")
    return match


def parse_cognitive(val):
    if val is None or str(val).strip() == "":
        raise ValueError("Missing required field: Cognitive Function Score")
    try:
        score = float(val)
    except (ValueError, TypeError):
        raise ValueError(f"Non-numeric Cognitive Function Score: '{val}'. Must be a number between 30.0 and 100.0.")
    if np.isnan(score) or score < 30.0 or score > 100.0:
        raise ValueError(f"Invalid Cognitive Function Score: {score}. Must be between 30.0 and 100.0.")
    return score


# ============================
# PREDICT ROUTE
# ============================
@app.route("/submit", methods=["POST"])
def submit():
    if not models_ready():
        return "Models not loaded", 500

    try:
        kfr = parse_binary(request.form.get("inputKFR"), "Kayser-Fleischer Rings")
        psychiatric = parse_binary(request.form.get("inputPsychiatric"), "Psychiatric Symptoms")
        family_history = parse_binary(request.form.get("inputFamilyHistory"), "Family History")
        gene_mutation = parse_binary(request.form.get("inputGeneMutation"), "ATB7B Gene Mutation")
        region = parse_region(request.form.get("inputRegion"))
        cognitive_score = parse_cognitive(request.form.get("inputCognitive"))

        cols = [
            "Age","Sex","Ceruloplasmin Level","Copper in Blood Serum",
            "Free Copper in Blood Serum","Copper in Urine","ALT","AST",
            "Total Bilirubin","Albumin","Alkaline Phosphatase (ALP)",
            "Prothrombin Time / INR","Gamma-Glutamyl Transferase (GGT)",
            "Kayser-Fleischer Rings","Neurological Symptoms Score",
            "Psychiatric Symptoms","Cognitive Function Score",
            "Family History","ATB7B Gene Mutation","Region",
            "Socioeconomic Status","Alcohol Use","BMI"
        ]

        values = [
            float(request.form.get("inputAge")),
            request.form.get("inputGender"),
            float(request.form.get("inputCeruloplasmin")),
            float(request.form.get("inputCopperBlood")),
            float(request.form.get("inputFreeCopperBlood")),
            float(request.form.get("inputCopperUrine")),
            float(request.form.get("inputALT")),
            float(request.form.get("inputAST")),
            float(request.form.get("inputTotalBilirubin")),
            float(request.form.get("inputAlbumin")),
            float(request.form.get("inputALP")),
            float(request.form.get("inputProthrombin")),
            float(request.form.get("inputGGT")),
            kfr,
            float(request.form.get("inputNeurological")),
            psychiatric,
            cognitive_score,
            family_history,
            gene_mutation,
            region,
            request.form.get("inputSocioeconomicStatus"),
            request.form.get("inputAlcoholUse"),
            float(request.form.get("inputBMI"))
        ]

        df = pd.DataFrame([values], columns=cols)

        prob, pred, shap_res, used_syn, used_df, model_breakdown = safe_shap_and_predict(df)

        txt = "Positive – Wilson Disease Detected" if pred == 1 else "Negative – No Wilson Disease"

        # Generate RAG clinical recommendations with 100% authentic patient data
        patient_dict = used_df.iloc[0].to_dict()
        rag_advice = rag_assistant.get_clinical_recommendations(patient_dict, txt)

        return render_template(
            "result.html",
            prediction_text=txt,
            probability=round(prob, 4),
            shap_waterfall=shap_res.get("waterfall"),
            data=patient_dict,
            used_synthetic=used_syn,
            clinical_advice=rag_advice,
            model_breakdown=model_breakdown
        )
    except (ValueError, TypeError) as ve:
        logger.warning("Validation error in /submit: %s", ve)
        return str(ve), 400
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return str(e), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        req_data = request.get_json(silent=True) or {}
        user_msg = req_data.get("message", "").strip()
        patient_context = req_data.get("patient_context", None)
        chat_history = req_data.get("chat_history", [])
        if not user_msg:
            return jsonify({"response": "Please enter a valid question."}), 400
        reply = rag_assistant.ask_chatbot(user_msg, patient_context=patient_context, chat_history=chat_history)
        return jsonify({"response": reply})
    except Exception as e:
        logger.exception("Chat API error: %s", e)
        return jsonify({"response": f"Error processing query: {str(e)}"}), 500


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
