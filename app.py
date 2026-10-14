# app.py — Advanced SHAP UI dashboard + prediction endpoints (FINAL FIXED VERSION)
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Response, session, redirect
import os
import sys
import time

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
from typing import Optional, Dict, Any, List

from rag_system import MedicalRAGSystem
from clinical_evaluation import (
    evaluate_clinical_parameters,
    calculate_patient_leipzig_breakdown,
    generate_physician_next_steps
)
from database import (
    init_db,
    get_next_patient_id,
    create_or_update_patient,
    get_patient_by_id,
    get_all_patients,
    save_assessment,
    get_assessment_by_id,
    get_patient_history,
    verify_user_credentials,
    change_user_password,
    log_audit_event,
    get_user_by_id,
    DEFAULT_DB_PATH
)
from lab_import import (
    parse_lab_file_content,
    generate_sample_csv_text,
    generate_sample_json_text,
    generate_sample_docx_bytes,
    MAX_UPLOAD_SIZE
)
from auth_middleware import (
    init_auth_middleware,
    login_required,
    roles_required,
    check_login_rate_limit,
    record_failed_login,
    clear_failed_logins,
    get_client_ip,
    generate_csrf_token,
    validate_csrf_token,
    is_safe_redirect_url
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wilson_app")

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Production Security Configuration: Fail-Safe Secret Key & Secure Cookie Enforcement
DEV_SECRET_KEY = "wilson-neuromed-clinical-dev-secret-key-change-in-prod"
is_production = os.environ.get("FLASK_ENV", "").lower() == "production" or os.environ.get("ENV", "").lower() == "production"
raw_secret = os.environ.get("FLASK_SECRET_KEY", "").strip()

if is_production:
    if not raw_secret or raw_secret == DEV_SECRET_KEY or len(raw_secret) < 32:
        raise RuntimeError(
            "Production security check failed: FLASK_SECRET_KEY must be explicitly configured as a high-entropy "
            "secret (at least 32 characters) and cannot be empty, weak, or the default development placeholder."
        )
    app.config['SECRET_KEY'] = raw_secret
    app.config['SESSION_COOKIE_SECURE'] = True
else:
    app.config['SECRET_KEY'] = raw_secret if raw_secret else DEV_SECRET_KEY
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() in ("true", "1")

app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['DB_PATH'] = os.environ.get("WILSON_DB_PATH", DEFAULT_DB_PATH)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Register Authentication, RBAC, CSRF, and Session Hardening Middleware
init_auth_middleware(app)


@app.after_request
def add_security_headers(response):
    """Adds defensive HTTP security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

def get_active_db_path():
    return app.config.get("DB_PATH", os.environ.get("WILSON_DB_PATH", DEFAULT_DB_PATH))

# Safe database initialization on startup
try:
    init_db(get_active_db_path())
except Exception as _db_err:
    logger.warning("Database initialization notice on startup: %s", _db_err)

# Initialize RAG System
rag_assistant = MedicalRAGSystem()

ARTIFACT_DIR = "wilson_artifacts"

preprocessor = None
svm = None
logreg = None
meta = None
calibrator = None
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
    calibrator = safe_load_artifact(os.path.join(ARTIFACT_DIR, "calibrator.joblib"))
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

    logger.info("Artifacts loaded: preproc=%s svm=%s logreg=%s meta=%s calibrator=%s bilstm=%s shap_bg=%s",
                preprocessor is not None, svm is not None, logreg is not None, meta is not None, calibrator is not None, bilstm is not None, shap_background is not None)
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

    if calibrator is not None:
        return calibrator.predict_proba(stacked)[:, 1]
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

    # Compute Bi-LSTM prediction if model is available
    if bilstm is not None:
        try:
            X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
            bilstm_p = float(bilstm.predict(X_lstm, verbose=0).ravel()[0])
            bilstm_prob = bilstm_p
        except Exception as e:
            logger.warning("Bi-LSTM prediction failed: %s", e)
            bilstm_p = (svm_p + log_p) / 2.0
            bilstm_prob = None
    else:
        bilstm_p = (svm_p + log_p) / 2.0
        bilstm_prob = None

    if getattr(meta, "n_features_in_", 2) == 3:
        stacked = np.column_stack([[bilstm_p], [svm_p], [log_p]])
    else:
        stacked = np.column_stack([[svm_p], [log_p]])

    if calibrator is not None:
        final_prob = float(calibrator.predict_proba(stacked)[0][1])
    else:
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


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Clinician and staff login endpoint.
    Includes rate-limiting (5 failures / 15m), audit logging, and session fixation protection.
    """
    if request.method == "GET":
        if "user_id" in session:
            role = session.get("role")
            if role == "clinician":
                return redirect(url_for("about"))
            elif role == "lab_staff":
                return redirect(url_for("api_sample_template"))
            elif role == "admin":
                return redirect(url_for("patient_dashboard"))
        next_url = request.args.get("next") or ""
        msg = request.args.get("message")
        return render_template("login.html", next_url=next_url, message=msg)

    # POST - Authenticate
    if request.is_json:
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        next_url = data.get("next") or ""
    else:
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        next_url = (request.form.get("next") or request.args.get("next") or "").strip()

    client_ip = get_client_ip()
    db_path = get_active_db_path()

    # 1. Rate limiting check
    allowed, remaining_sec = check_login_rate_limit(client_ip, username)
    if not allowed:
        log_audit_event(
            action="LOGIN_RATE_LIMITED",
            status="DENIED",
            username=username,
            ip_address=client_ip,
            details=f"Rate limit exceeded: locked for {remaining_sec} seconds",
            db_path=db_path
        )
        msg = f"Too many failed login attempts. Temporarily locked for {remaining_sec} seconds. Please try again later."
        if request.is_json:
            return jsonify({"success": False, "error": msg, "status": 429}), 429
        return render_template("login.html", error=msg, next_url=next_url), 429

    # 2. Check credentials
    user = verify_user_credentials(username, password, db_path=db_path)
    if not user:
        record_failed_login(client_ip, username)
        log_audit_event(
            action="LOGIN_FAILED",
            status="FAILURE",
            username=username,
            ip_address=client_ip,
            details="Invalid username or password, or account inactive",
            db_path=db_path
        )
        msg = "Invalid username or password. Please verify your credentials and try again."
        if request.is_json:
            return jsonify({"success": False, "error": msg, "status": 401}), 401
        return render_template("login.html", error=msg, next_url=next_url), 401

    # 3. Successful authentication: clear rate limiter and regenerate session
    clear_failed_logins(client_ip, username)
    session.clear()
    session["user_id"] = user["user_id"]
    session["username"] = user["username"]
    session["role"] = user["role"]
    session["full_name"] = user["full_name"]
    session["last_active"] = time.time()
    generate_csrf_token()

    log_audit_event(
        action="LOGIN_SUCCESS",
        status="SUCCESS",
        username=user["username"],
        user_id=user["user_id"],
        ip_address=client_ip,
        details=f"User authenticated successfully with role '{user['role']}'",
        db_path=db_path
    )

    if request.is_json:
        return jsonify({"success": True, "user": user, "message": "Authenticated successfully"}), 200

    # Direct to mandatory password change if flag is set
    if user.get("must_change_password") == 1:
        return redirect(url_for("change_password"))

    # Validate next_url to prevent open redirect attacks
    if next_url and is_safe_redirect_url(next_url):
        return redirect(next_url)

    if user["role"] == "clinician":
        return redirect(url_for("about"))
    elif user["role"] == "lab_staff":
        return redirect(url_for("lab_import_page"))
    return redirect(url_for("patient_dashboard"))


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    """Terminates session and records logout audit log. POST-only and CSRF protected."""
    db_path = get_active_db_path()
    username = session.get("username")
    user_id = session.get("user_id")
    client_ip = get_client_ip()

    log_audit_event(
        action="LOGOUT",
        status="SUCCESS",
        username=username,
        user_id=user_id,
        ip_address=client_ip,
        details="User logged out and session cleared",
        db_path=db_path
    )
    session.clear()

    if request.is_json:
        return jsonify({"success": True, "message": "Signed out successfully."}), 200

    return redirect(url_for("login", message="You have been signed out successfully."))


@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    """Enforces mandatory or voluntary password changes server-side with CSRF protection."""
    db_path = get_active_db_path()
    user_id = session.get("user_id")
    username = session.get("username")
    client_ip = get_client_ip()

    if request.method == "POST":
        if request.is_json:
            data = request.get_json(silent=True) or {}
            old_pw = data.get("current_password", "")
            new_pw = data.get("new_password", "")
            confirm_pw = data.get("confirm_password", "")
        else:
            old_pw = request.form.get("current_password", "")
            new_pw = request.form.get("new_password", "")
            confirm_pw = request.form.get("confirm_password", "")

        if not old_pw or not new_pw:
            err = "Both current and new passwords are required."
            if request.is_json:
                return jsonify({"success": False, "error": err}), 400
            return render_template("change_password.html", error=err), 400

        if new_pw != confirm_pw:
            err = "New passwords do not match."
            if request.is_json:
                return jsonify({"success": False, "error": err}), 400
            return render_template("change_password.html", error=err), 400

        success, msg = change_user_password(
            user_id=user_id,
            old_password=old_pw,
            new_password=new_pw,
            db_path=db_path
        )
        if not success:
            log_audit_event(
                action="PASSWORD_CHANGE_FAILED",
                status="DENIED",
                username=username,
                user_id=user_id,
                ip_address=client_ip,
                details="Password change rejected by validation policy",
                db_path=db_path
            )
            if request.is_json:
                return jsonify({"success": False, "error": msg}), 400
            return render_template("change_password.html", error=msg), 400

        log_audit_event(
            action="PASSWORD_CHANGED",
            status="SUCCESS",
            username=username,
            user_id=user_id,
            ip_address=client_ip,
            details="User updated password successfully and cleared must_change_password flag",
            db_path=db_path
        )

        if request.is_json:
            return jsonify({"success": True, "message": "Password changed successfully."}), 200

        role = session.get("role")
        if role == "clinician":
            return redirect(url_for("about"))
        elif role == "lab_staff":
            return redirect(url_for("lab_import_page"))
        return redirect(url_for("patient_dashboard"))

    return render_template("change_password.html")


@app.route("/lab/import", methods=["GET"])
@roles_required("lab_staff", "clinician")
def lab_import_page():
    """Laboratory test file import and schema validation portal for laboratory personnel."""
    return render_template("lab_import.html")


@app.route("/about")
@roles_required("clinician")
def about():
    try:
        next_id = get_next_patient_id(get_active_db_path())
        all_patients = get_all_patients(get_active_db_path())
    except Exception as e:
        logger.warning("Error fetching patient preview for /about: %s", e)
        next_id = "NEUROMED001"
        all_patients = []
    return render_template("about.html", next_patient_id=next_id, patients=all_patients)


@app.route("/api/next_patient_id", methods=["GET"])
@roles_required("clinician")
def api_next_patient_id():
    """Returns the next sequential Patient ID preview (e.g. NEUROMED001)."""
    try:
        next_id = get_next_patient_id(get_active_db_path())
        return jsonify({"success": True, "next_patient_id": next_id}), 200
    except Exception as e:
        logger.warning("API next_patient_id error: %s", e)
        return jsonify({"success": False, "error": "Failed to determine next patient ID"}), 500


@app.route("/api/patient/<patient_id>", methods=["GET"])
@roles_required("clinician")
def api_get_patient(patient_id):
    """Retrieves patient demographic profile by ID without exposing sensitive internal data."""
    try:
        p = get_patient_by_id(patient_id, get_active_db_path())
        if not p:
            return jsonify({"success": False, "error": f"Patient '{patient_id}' not found"}), 404
        return jsonify({
            "success": True,
            "patient": {
                "patient_id": p["patient_id"],
                "patient_name": p["patient_name"],
                "age": p["age"],
                "sex": p["sex"],
                "region": p["region"],
                "ses": p["ses"],
                "bmi": p["bmi"],
                "alcohol_use": p["alcohol_use"]
            }
        }), 200
    except Exception as e:
        logger.warning("API get_patient error for %s: %s", patient_id, e)
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/upload_lab_file", methods=["POST"])
@roles_required("lab_staff", "clinician")
def api_upload_lab_file():
    """
    Parses and validates uploaded CSV or JSON laboratory data files.
    Enforces 2 MB size limit, schema validation, and missing value preservation.
    Does NOT execute code or run prediction.
    """
    try:
        # Check Content-Length header if present to short-circuit oversized payloads
        content_length = request.content_length
        if content_length is not None and content_length > MAX_UPLOAD_SIZE:
            return jsonify({
                "success": False,
                "error": f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
            }), 413

        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded. Please select a .csv, .json, .pdf, or Word laboratory report."}), 400

        uploaded = request.files["file"]
        if not uploaded or not uploaded.filename:
            return jsonify({"success": False, "error": "No file selected or empty filename."}), 400

        filename = uploaded.filename
        file_bytes = uploaded.read(MAX_UPLOAD_SIZE + 1)
        if len(file_bytes) > MAX_UPLOAD_SIZE:
            return jsonify({
                "success": False,
                "error": f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
            }), 413

        parsed = parse_lab_file_content(file_bytes, filename)
        status_code = 200 if parsed.get("success") else 400
        return jsonify(parsed), status_code
    except Exception as e:
        logger.exception("Error processing laboratory file upload: %s", e)
        return jsonify({"success": False, "error": f"Internal server error parsing file: {str(e)}"}), 500


@app.route("/api/sample_template", methods=["GET"])
@roles_required("lab_staff", "clinician")
def api_sample_template():
    """Returns downloadable sample CSV, JSON, or Word DOCX laboratory data template."""
    fmt = request.args.get("format", "csv").lower().strip()
    if fmt == "json":
        json_content = generate_sample_json_text()
        return Response(
            json_content,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=wilson_lab_template.json"}
        )
    if fmt == "docx":
        docx_bytes = generate_sample_docx_bytes()
        return Response(
            docx_bytes,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=wilson_lab_template.docx"}
        )
    csv_content = generate_sample_csv_text()
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=wilson_lab_template.csv"}
    )



def safe_shap_path(rel_path: Optional[str]) -> Optional[str]:
    """
    Validates that a SHAP image path is secure (no directory traversal)
    and that the file actually exists on disk inside static/.
    """
    if not rel_path or not isinstance(rel_path, str):
        return None
    cleaned = rel_path.replace("\\", "/").strip().lstrip("/")
    if ".." in cleaned or ":" in cleaned:
        return None
    if cleaned.startswith("static/"):
        cleaned = cleaned[len("static/"):]
    full_path = os.path.join("static", cleaned.replace("/", os.sep))
    if os.path.isfile(full_path):
        return cleaned
    return None


@app.route("/patients", methods=["GET"])
@roles_required("clinician")
def patient_dashboard():
    """Displays patient directory dashboard with search, risk filters, sorting, and pagination."""
    try:
        search_query = request.args.get("search", "").strip()
        risk_filter = request.args.get("risk", "all").strip().lower()
        sort_by = request.args.get("sort", "created_desc").strip().lower()
        
        try:
            page = max(1, int(request.args.get("page", 1)))
        except (ValueError, TypeError):
            page = 1
        per_page = 15

        all_matches = get_all_patients(
            get_active_db_path(),
            search=search_query if search_query else None,
            risk_filter=risk_filter if risk_filter != "all" else None,
            sort_by=sort_by
        )
        total_count = len(all_matches)
        total_pages = max(1, (total_count + per_page - 1) // per_page)
        page = min(page, total_pages)
        
        start_idx = (page - 1) * per_page
        patients_page = all_matches[start_idx : start_idx + per_page]

        return render_template(
            "patients.html",
            patients=patients_page,
            search=search_query,
            risk_filter=risk_filter,
            sort_by=sort_by,
            page=page,
            total_pages=total_pages,
            total_count=total_count
        )
    except Exception as e:
        logger.exception("Error loading patient directory: %s", e)
        return "Failed to load patient records", 500


@app.route("/patient/<patient_id>/history", methods=["GET"])
@roles_required("clinician")
def patient_history_view(patient_id):
    """
    Displays full chronological assessment history for an individual patient.
    Pure database query; does not execute ML model.
    """
    try:
        db_path = get_active_db_path()
        patient = get_patient_by_id(patient_id, db_path)
        if not patient:
            return render_template(
                "404.html",
                message=f"Patient record '{patient_id}' was not found in the database."
            ), 404
        
        history = get_patient_history(patient_id, db_path)
        return render_template("patient_history.html", patient=patient, history=history)
    except Exception as e:
        logger.exception("Error retrieving patient history for %s: %s", patient_id, e)
        return "Failed to load patient history", 500


@app.route("/assessment/<int:assessment_id>/report", methods=["GET"])
@roles_required("clinician")
def historical_assessment_report(assessment_id):
    """
    Renders an archival clinical decision-support report for a specific historical assessment.
    Reconstructs clinical metrics and Leipzig breakdown deterministically without re-running ML models.
    """
    try:
        db_path = get_active_db_path()
        assessment = get_assessment_by_id(assessment_id, db_path)
        if not assessment:
            return render_template(
                "404.html",
                message=f"Assessment record #{assessment_id} was not found in the database."
            ), 404

        inputs_dict = assessment.get("inputs_dict") or {}
        clinical_table = evaluate_clinical_parameters(inputs_dict)
        leipzig_breakdown = calculate_patient_leipzig_breakdown(inputs_dict)
        
        pred_label = assessment.get("prediction_label", 0)
        prob = assessment.get("probability", 0.0) or 0.0
        leipzig_total = leipzig_breakdown.get("total_score", 0)
        physician_steps = generate_physician_next_steps(pred_label, prob, inputs_dict, leipzig_total)

        # Model breakdown
        model_breakdown = {
            "svm": round(assessment["svm_prob"] * 100, 1) if assessment.get("svm_prob") is not None else None,
            "logreg": round(assessment["logreg_prob"] * 100, 1) if assessment.get("logreg_prob") is not None else None,
            "bilstm": round(assessment["bilstm_prob"] * 100, 1) if assessment.get("bilstm_prob") is not None else None
        }

        # Safe SHAP path check (prevents directory traversal and checks file existence)
        safe_shap = safe_shap_path(assessment.get("shap_waterfall_path"))

        pred_text = assessment.get("prediction_text")
        if not pred_text:
            pred_text = "High Predicted Risk – Wilson Disease" if pred_label == 1 else "Low Predicted Risk – Wilson Disease"

        # Format assessment date nicely
        raw_date = assessment.get("created_at") or ""
        formatted_date = raw_date
        try:
            dt = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
            formatted_date = dt.strftime("%B %d, %Y — %H:%M UTC")
        except Exception:
            pass

        return render_template(
            "result.html",
            prediction_text=pred_text,
            probability=round(prob, 4),
            shap_waterfall=safe_shap,
            data=inputs_dict,
            clinical_table=clinical_table,
            leipzig_breakdown=leipzig_breakdown,
            physician_steps=physician_steps,
            used_synthetic=False,
            clinical_advice=None,
            model_breakdown=model_breakdown,
            patient_id=assessment["patient_id"],
            patient_name=assessment.get("patient_name", "Anonymous Patient"),
            assessment_id=assessment["assessment_id"],
            assessment_date=formatted_date,
            is_historical=True,
            stored_leipzig_score=assessment.get("leipzig_score"),
            model_version=assessment.get("model_version") or "v1.0.0"
        )
    except Exception as e:
        logger.exception("Error rendering historical assessment report #%s: %s", assessment_id, e)
        return "Failed to load historical assessment report", 500


@app.route("/do")
@roles_required("clinician")
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
@roles_required("clinician")
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

        # 1. Resolve or Create Patient Record in Database
        db_path = get_active_db_path()
        patient_name = (request.form.get("patientName") or "Anonymous Patient").strip()
        patient_id = (request.form.get("patientId") or "").strip()

        existing_patient = get_patient_by_id(patient_id, db_path) if patient_id else None
        if existing_patient:
            # Reassessment of existing patient - preserve their saved baseline demographics
            patient_record = existing_patient
        else:
            # New patient record with allocated sequential ID
            patient_record = create_or_update_patient({
                "patient_id": patient_id if patient_id else None,
                "patient_name": patient_name,
                "Age": values[0],
                "Sex": values[1],
                "Region": region,
                "Socioeconomic Status": values[20],
                "Alcohol Use": values[21],
                "BMI": values[22]
            }, db_path=db_path)

        # 2. Execute Multi-Model Prediction Pipeline (Unchanged 23 features)
        prob, pred, shap_res, used_syn, used_df, model_breakdown = safe_shap_and_predict(df)

        txt = "High Predicted Risk – Wilson Disease" if pred == 1 else "Low Predicted Risk – Wilson Disease"

        # Generate RAG clinical recommendations with 100% authentic patient data
        patient_dict = used_df.iloc[0].to_dict()
        rag_advice = rag_assistant.get_clinical_recommendations(patient_dict, txt)

        # Dynamic Patient-Based Clinical Evaluation & Interpretation (Issue-7)
        clinical_table = evaluate_clinical_parameters(patient_dict)
        leipzig_breakdown = calculate_patient_leipzig_breakdown(patient_dict)
        physician_steps = generate_physician_next_steps(pred, prob, patient_dict, leipzig_breakdown["total_score"])

        # 3. Persist Assessment to Database (Only after prediction & clinical scoring succeed)
        model_ver = "v2.0.0-calibrated"
        assessment_results = {
            "prediction_text": txt,
            "prediction_label": pred,
            "probability": round(prob, 4),
            "model_breakdown": model_breakdown,
            "leipzig_score": leipzig_breakdown.get("total_score", 0),
            "shap_waterfall_path": shap_res.get("waterfall"),
            "model_version": model_ver
        }
        assessment_id = save_assessment(
            patient_id=patient_record["patient_id"],
            clinical_inputs=patient_dict,
            prediction_results=assessment_results,
            db_path=db_path
        )

        return render_template(
            "result.html",
            prediction_text=txt,
            probability=round(prob, 4),
            shap_waterfall=shap_res.get("waterfall"),
            data=patient_dict,
            clinical_table=clinical_table,
            leipzig_breakdown=leipzig_breakdown,
            physician_steps=physician_steps,
            used_synthetic=used_syn,
            clinical_advice=rag_advice,
            model_breakdown=model_breakdown,
            patient_id=patient_record["patient_id"],
            patient_name=patient_record["patient_name"],
            assessment_id=assessment_id,
            assessment_date=datetime.now().strftime("%B %d, %Y — %H:%M"),
            is_historical=False,
            stored_leipzig_score=leipzig_breakdown.get("total_score", 0),
            model_version=model_ver
        )
    except (ValueError, TypeError) as ve:
        logger.warning("Validation error in /submit: %s", ve)
        return str(ve), 400
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return "An internal server error occurred while processing the clinical assessment. Please check server logs.", 500


@app.route("/api/chat", methods=["POST"])
@roles_required("clinician")
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
        return jsonify({"response": "An error occurred while processing your clinical query. Please try again or consult server logs."}), 500


@app.route("/uploads/<path:filename>")
@roles_required("clinician")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ============================
# ERROR HANDLERS
# ============================
@app.errorhandler(403)
def handle_forbidden(err):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"success": False, "error": "Access Forbidden: You do not have permission to access this resource", "status": 403}), 403
    return render_template("403.html", message="Access Forbidden: You do not have permission to access this resource."), 403


@app.errorhandler(404)
def handle_not_found(err):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"success": False, "error": "Resource Not Found", "status": 404}), 404
    return render_template("404.html", message="The requested clinical resource or page was not found."), 404


@app.errorhandler(405)
def handle_method_not_allowed(err):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"success": False, "error": "Method Not Allowed", "status": 405}), 405
    return render_template("403.html", message="HTTP Method Not Allowed for this endpoint."), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
