"""
database.py

Patient Database & Assessment History Layer for NeuroMed AI.
Provides:
1. Normalized SQLite persistence for patient demographics and assessment history.
2. Thread-safe, concurrency-guarded sequential Patient ID generation (e.g. NEUROMED001, NEUROMED002).
3. Preservation of all 23 clinical inputs in their exact canonical definitions.
4. Strict preservation of missing values as None (SQL NULL) without zero-fabrication.
5. Foreign key constraint enforcement (PRAGMA foreign_keys = ON) and transaction rollbacks on failure.
6. Reusable for both production runtime and isolated test environments via configurable db_path.
"""

import os
import sqlite3
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger("wilson_database")

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wilson_records.db")

# Canonical mapping of the exact 23 clinical features to assessment table columns
FEATURE_TO_COL_MAP = {
    "Age": "age",
    "Sex": "sex",
    "Ceruloplasmin Level": "ceruloplasmin_level",
    "Copper in Blood Serum": "copper_blood_serum",
    "Free Copper in Blood Serum": "free_copper_blood_serum",
    "Copper in Urine": "copper_urine",
    "ALT": "alt",
    "AST": "ast",
    "Total Bilirubin": "total_bilirubin",
    "Albumin": "albumin",
    "Alkaline Phosphatase (ALP)": "alkaline_phosphatase_alp",
    "Prothrombin Time / INR": "prothrombin_time_inr",
    "Gamma-Glutamyl Transferase (GGT)": "gamma_glutamyl_transferase_ggt",
    "Kayser-Fleischer Rings": "kayser_fleischer_rings",
    "Neurological Symptoms Score": "neurological_symptoms_score",
    "Psychiatric Symptoms": "psychiatric_symptoms",
    "Cognitive Function Score": "cognitive_function_score",
    "Family History": "family_history",
    "ATB7B Gene Mutation": "atb7b_gene_mutation",
    "Region": "region",
    "Socioeconomic Status": "socioeconomic_status",
    "Alcohol Use": "alcohol_use",
    "BMI": "bmi"
}

COL_TO_FEATURE_MAP = {v: k for k, v in FEATURE_TO_COL_MAP.items()}


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Always enforces PRAGMA foreign_keys = ON and enables dictionary-like Row access.
    """
    target_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(target_path, timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def open_db(db_path: Optional[str] = None):
    """Context manager that guarantees the SQLite connection is closed upon exit."""
    conn = get_db_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes the database schema if tables do not exist.
    Creates patients and assessments tables along with performance indexes.
    """
    target_path = db_path or DEFAULT_DB_PATH
    db_dir = os.path.dirname(os.path.abspath(target_path))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    with open_db(target_path) as conn:
        cursor = conn.cursor()

        # 1. Patients Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                patient_name TEXT NOT NULL,
                age REAL,
                sex TEXT,
                region TEXT,
                ses TEXT,
                bmi REAL,
                alcohol_use TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Assessments Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Exact 23 Clinical Input Parameters
                age REAL,
                sex TEXT,
                ceruloplasmin_level REAL,
                copper_blood_serum REAL,
                free_copper_blood_serum REAL,
                copper_urine REAL,
                alt REAL,
                ast REAL,
                total_bilirubin REAL,
                albumin REAL,
                alkaline_phosphatase_alp REAL,
                prothrombin_time_inr REAL,
                gamma_glutamyl_transferase_ggt REAL,
                kayser_fleischer_rings INTEGER,
                neurological_symptoms_score REAL,
                psychiatric_symptoms INTEGER,
                cognitive_function_score REAL,
                family_history INTEGER,
                atb7b_gene_mutation INTEGER,
                region TEXT,
                socioeconomic_status TEXT,
                alcohol_use TEXT,
                bmi REAL,
                
                -- Model Prediction and Clinical Evaluation Outputs
                prediction_text TEXT,
                prediction_label INTEGER,
                probability REAL,
                bilstm_prob REAL,
                svm_prob REAL,
                logreg_prob REAL,
                leipzig_score INTEGER,
                shap_waterfall_path TEXT,
                model_version TEXT DEFAULT 'v2.0.0-calibrated',
                
                -- Structured Inputs Snapshot (JSON)
                inputs_json TEXT,
                
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
            );
        """)

        # Schema migration check: ensure model_version column exists on existing assessments tables
        cursor.execute("PRAGMA table_info(assessments);")
        existing_cols = [row[1] for row in cursor.fetchall()]
        if existing_cols and "model_version" not in existing_cols:
            cursor.execute("ALTER TABLE assessments ADD COLUMN model_version TEXT DEFAULT 'v1.0.0';")

        # Indexes for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assessments_patient_id ON assessments(patient_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assessments_created_at ON assessments(created_at);")

        # 3. Users Table (RBAC Authentication)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'clinician', 'lab_staff')),
                is_active INTEGER NOT NULL DEFAULT 1,
                must_change_password INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")

        # 4. Security Audit Log Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                action TEXT NOT NULL,
                resource_id TEXT,
                ip_address TEXT,
                status TEXT NOT NULL,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_logs(created_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_logs(user_id);")

        # 5. RAG Document Governance Table (Chief Doctor Authorization Gatekeeper)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                original_filename TEXT NOT NULL,
                title TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                file_hash_sha256 TEXT NOT NULL,
                uploaded_by_id INTEGER,
                uploaded_by_name TEXT,
                review_status TEXT NOT NULL DEFAULT 'pending_approval' CHECK(review_status IN ('pending_approval', 'approved', 'rejected')),
                reviewed_by_id INTEGER,
                reviewed_by_name TEXT,
                review_notes TEXT,
                passages_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                FOREIGN KEY (uploaded_by_id) REFERENCES users(user_id) ON DELETE SET NULL,
                FOREIGN KEY (reviewed_by_id) REFERENCES users(user_id) ON DELETE SET NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_status ON rag_documents(review_status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_filename ON rag_documents(filename);")

        # 6. Persistent Chat Sessions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                patient_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at);")

        # 7. Turn-by-Turn Chat Messages Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                citations_json TEXT,
                synthesis_engine TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);")

        # 8. RAG Query Telemetry & Audit Logs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_query_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                session_id TEXT,
                query_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                synthesis_engine TEXT NOT NULL,
                citation_valid INTEGER NOT NULL DEFAULT 1,
                fallback_reason TEXT,
                retrieved_passages_count INTEGER DEFAULT 0,
                sources_cited_json TEXT,
                latency_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE SET NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_query_logs_created_at ON rag_query_logs(created_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_query_logs_engine ON rag_query_logs(synthesis_engine);")

        conn.commit()
    logger.info("Database initialized successfully at: %s", target_path)


def _compute_next_id_from_cursor(cursor: sqlite3.Cursor, prefix: str = "NEUROMED") -> str:
    """Helper to compute the next sequential formatted ID within an active cursor."""
    cursor.execute("SELECT patient_id FROM patients WHERE patient_id LIKE ?", (f"{prefix}%",))
    rows = cursor.fetchall()
    max_num = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for r in rows:
        pid = r["patient_id"] if isinstance(r, sqlite3.Row) else r[0]
        match = pattern.match(pid)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    next_num = max_num + 1
    width = max(3, len(str(next_num)))
    return f"{prefix}{next_num:0{width}d}"


def get_next_patient_id(db_path: Optional[str] = None, prefix: str = "NEUROMED") -> str:
    """
    Returns the next sequential patient ID (e.g. NEUROMED001).
    Does not allocate or lock the ID; intended for UI display preview.
    """
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        return _compute_next_id_from_cursor(cursor, prefix=prefix)


def create_or_update_patient(
    patient_data: Dict[str, Any],
    db_path: Optional[str] = None,
    prefix: str = "NEUROMED"
) -> Dict[str, Any]:
    """
    Creates a new patient or updates an existing patient's demographic profile.
    If patient_id is not provided or set to 'auto', safely generates the next sequential ID
    within a transaction, retrying in case of collision.
    """
    name = (patient_data.get("patient_name") or "").strip()
    if not name:
        raise ValueError("Patient Name is required and cannot be empty.")

    requested_id = (patient_data.get("patient_id") or "").strip()
    age = patient_data.get("Age") if "Age" in patient_data else patient_data.get("age")
    sex = patient_data.get("Sex") if "Sex" in patient_data else patient_data.get("sex")
    region = patient_data.get("Region") if "Region" in patient_data else patient_data.get("region")
    ses = patient_data.get("Socioeconomic Status") if "Socioeconomic Status" in patient_data else patient_data.get("ses")
    bmi = patient_data.get("BMI") if "BMI" in patient_data else patient_data.get("bmi")
    alcohol = patient_data.get("Alcohol Use") if "Alcohol Use" in patient_data else patient_data.get("alcohol_use")

    target_path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(target_path)

    try:
        cursor = conn.cursor()
        # If specific ID was requested, check if it already exists
        if requested_id and requested_id.upper() != "AUTO":
            cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (requested_id,))
            existing = cursor.fetchone()
            if existing:
                # Update demographic info
                cursor.execute("""
                    UPDATE patients
                    SET patient_name = ?,
                        age = COALESCE(?, age),
                        sex = COALESCE(?, sex),
                        region = COALESCE(?, region),
                        ses = COALESCE(?, ses),
                        bmi = COALESCE(?, bmi),
                        alcohol_use = COALESCE(?, alcohol_use),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE patient_id = ?;
                """, (name, age, sex, region, ses, bmi, alcohol, requested_id))
                conn.commit()
                return get_patient_by_id(requested_id, db_path=target_path)
            else:
                # Insert with requested ID
                cursor.execute("""
                    INSERT INTO patients (patient_id, patient_name, age, sex, region, ses, bmi, alcohol_use)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (requested_id, name, age, sex, region, ses, bmi, alcohol))
                conn.commit()
                return get_patient_by_id(requested_id, db_path=target_path)

        # Auto-generation loop with retry to guarantee concurrency safety
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                cursor.execute("BEGIN IMMEDIATE;")
                new_id = _compute_next_id_from_cursor(cursor, prefix=prefix)
                cursor.execute("""
                    INSERT INTO patients (patient_id, patient_name, age, sex, region, ses, bmi, alcohol_use)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (new_id, name, age, sex, region, ses, bmi, alcohol))
                conn.commit()
                return get_patient_by_id(new_id, db_path=target_path)
            except sqlite3.IntegrityError:
                conn.rollback()
                if attempt == max_attempts - 1:
                    raise
        raise RuntimeError("Failed to allocate a unique patient ID after multiple attempts.")
    finally:
        conn.close()


def get_patient_by_id(patient_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single patient record by ID, or None if not found."""
    if not patient_id:
        return None
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id.strip(),))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def get_all_patients(
    db_path: Optional[str] = None,
    search: Optional[str] = None,
    risk_filter: Optional[str] = None,
    sort_by: Optional[str] = "created_desc",
    page: Optional[int] = None,
    per_page: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves all patients with summary statistics of their most recent assessment.
    Supports search by patient ID / name, filtering by prediction risk category,
    sorting, and optional pagination.
    """
    base_query = """
        WITH patient_summaries AS (
            SELECT 
                p.patient_id,
                p.patient_name,
                p.age,
                p.sex,
                p.region,
                p.bmi,
                p.created_at,
                COUNT(a.assessment_id) as assessment_count,
                MAX(a.created_at) as latest_assessment_date,
                (
                    SELECT a2.assessment_id 
                    FROM assessments a2 
                    WHERE a2.patient_id = p.patient_id 
                    ORDER BY a2.created_at DESC, a2.assessment_id DESC 
                    LIMIT 1
                ) as latest_assessment_id,
                (
                    SELECT a2.prediction_text 
                    FROM assessments a2 
                    WHERE a2.patient_id = p.patient_id 
                    ORDER BY a2.created_at DESC, a2.assessment_id DESC 
                    LIMIT 1
                ) as latest_prediction_text,
                (
                    SELECT a2.probability 
                    FROM assessments a2 
                    WHERE a2.patient_id = p.patient_id 
                    ORDER BY a2.created_at DESC, a2.assessment_id DESC 
                    LIMIT 1
                ) as latest_probability,
                (
                    SELECT a2.leipzig_score 
                    FROM assessments a2 
                    WHERE a2.patient_id = p.patient_id 
                    ORDER BY a2.created_at DESC, a2.assessment_id DESC 
                    LIMIT 1
                ) as latest_leipzig_score,
                (
                    SELECT a2.model_version 
                    FROM assessments a2 
                    WHERE a2.patient_id = p.patient_id 
                    ORDER BY a2.created_at DESC, a2.assessment_id DESC 
                    LIMIT 1
                ) as latest_model_version
            FROM patients p
            LEFT JOIN assessments a ON p.patient_id = a.patient_id
            GROUP BY p.patient_id
        )
        SELECT * FROM patient_summaries
    """
    where_clauses = []
    params: List[Any] = []

    if search and search.strip():
        term = f"%{search.strip()}%"
        where_clauses.append("(patient_id LIKE ? OR patient_name LIKE ?)")
        params.extend([term, term])

    if risk_filter and risk_filter.strip().lower() not in ("all", ""):
        rf = risk_filter.strip().lower()
        if rf in ("high", "high_risk", "positive"):
            where_clauses.append("(latest_prediction_text LIKE '%High%' OR latest_probability >= 0.5)")
        elif rf in ("low", "low_risk", "negative"):
            where_clauses.append("(latest_prediction_text LIKE '%Low%' OR (latest_probability IS NOT NULL AND latest_probability < 0.5))")
        elif rf in ("unassessed", "none", "pending"):
            where_clauses.append("(assessment_count = 0)")

    query = base_query
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    # Sorting
    sort_key = (sort_by or "created_desc").lower().strip()
    if sort_key == "created_asc":
        query += " ORDER BY created_at ASC"
    elif sort_key == "name_asc":
        query += " ORDER BY patient_name COLLATE NOCASE ASC"
    elif sort_key == "name_desc":
        query += " ORDER BY patient_name COLLATE NOCASE DESC"
    elif sort_key == "risk_desc":
        query += " ORDER BY latest_probability DESC, created_at DESC"
    else:
        query += " ORDER BY created_at DESC"

    # Pagination
    if page is not None and per_page is not None and per_page > 0:
        offset = max(0, (page - 1) * per_page)
        query += " LIMIT ? OFFSET ?"
        params.extend([per_page, offset])

    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def save_assessment(
    patient_id: str,
    clinical_inputs: Dict[str, Any],
    prediction_results: Dict[str, Any],
    db_path: Optional[str] = None
) -> int:
    """
    Saves an assessment record linked via foreign key to patient_id.
    Strictly preserves missing values as None (SQL NULL) without zero fabrication.
    Rolls back transaction on any failure. Returns new assessment_id.
    """
    if not patient_id or not patient_id.strip():
        raise ValueError("Cannot save assessment: missing required patient_id.")
    if not isinstance(clinical_inputs, dict):
        raise ValueError("Cannot save assessment: clinical_inputs must be a dictionary.")

    pid = patient_id.strip()
    target_path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(target_path)

    try:
        cursor = conn.cursor()

        # Verify patient exists
        cursor.execute("SELECT patient_id FROM patients WHERE patient_id = ?", (pid,))
        if not cursor.fetchone():
            raise ValueError(f"Foreign key violation: Patient '{pid}' does not exist.")

        # Map the 23 clinical input fields
        cols = ["patient_id"]
        placeholders = ["?"]
        values = [pid]

        for feature_name, col_name in FEATURE_TO_COL_MAP.items():
            val = clinical_inputs.get(feature_name)
            # Clean empty strings into None
            if val is not None and str(val).strip() == "":
                val = None
            cols.append(col_name)
            placeholders.append("?")
            values.append(val)

        # Extract model outputs without fabricating any missing elements
        pred_text = prediction_results.get("prediction_text")
        pred_label = prediction_results.get("prediction_label")
        prob = prediction_results.get("probability")
        leipzig = prediction_results.get("leipzig_score")
        shap_path = prediction_results.get("shap_waterfall_path")

        mb = prediction_results.get("model_breakdown") or {}
        bilstm_prob = mb.get("bilstm")
        svm_prob = mb.get("svm")
        logreg_prob = mb.get("logreg")

        # Create structured JSON snapshot of inputs for exact auditability
        inputs_json = json.dumps(clinical_inputs, default=str)
        model_ver = prediction_results.get("model_version") or "v2.0.0-calibrated"

        cols.extend([
            "prediction_text", "prediction_label", "probability",
            "bilstm_prob", "svm_prob", "logreg_prob",
            "leipzig_score", "shap_waterfall_path", "inputs_json",
            "model_version"
        ])
        placeholders.extend(["?", "?", "?", "?", "?", "?", "?", "?", "?", "?"])
        values.extend([
            pred_text, pred_label, prob,
            bilstm_prob, svm_prob, logreg_prob,
            leipzig, shap_path, inputs_json,
            model_ver
        ])

        sql = f"INSERT INTO assessments ({', '.join(cols)}) VALUES ({', '.join(placeholders)});"
        cursor.execute(sql, values)
        new_assessment_id = cursor.lastrowid
        conn.commit()
        return new_assessment_id
    except Exception as e:
        conn.rollback()
        logger.warning("Failed to save assessment for patient %s: %s", pid, e)
        raise
    finally:
        conn.close()


def get_assessment_by_id(assessment_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an individual assessment record by assessment_id."""
    if not assessment_id:
        return None
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.*, p.patient_name 
            FROM assessments a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE a.assessment_id = ?;
        """, (assessment_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        if res.get("inputs_json"):
            try:
                res["inputs_dict"] = json.loads(res["inputs_json"])
            except Exception:
                res["inputs_dict"] = {}
        if not res.get("inputs_dict"):
            reconstructed = {}
            for col_name, feature_name in COL_TO_FEATURE_MAP.items():
                if col_name in res and res[col_name] is not None:
                    reconstructed[feature_name] = res[col_name]
            res["inputs_dict"] = reconstructed
        return res


def get_patient_history(patient_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all historical assessments for a patient, ordered by newest first.
    Never overwrites historical data.
    """
    if not patient_id:
        return []
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM assessments 
            WHERE patient_id = ? 
            ORDER BY created_at DESC, assessment_id DESC;
        """, (patient_id.strip(),))
        rows = cursor.fetchall()
        history = []
        for r in rows:
            d = dict(r)
            if d.get("inputs_json"):
                try:
                    d["inputs_dict"] = json.loads(d["inputs_json"])
                except Exception:
                    d["inputs_dict"] = {}
            if not d.get("inputs_dict"):
                reconstructed = {}
                for col_name, feature_name in COL_TO_FEATURE_MAP.items():
                    if col_name in d and d[col_name] is not None:
                        reconstructed[feature_name] = d[col_name]
                d["inputs_dict"] = reconstructed
            history.append(d)
        return history


# ==============================================================================
# DATABASE BACKUP HELPER (CRASH-CONSISTENT NATIVE SQLITE BACKUP)
# ==============================================================================

def backup_database(db_path: Optional[str] = None, target_backup_path: Optional[str] = None) -> str:
    """
    Creates an online, crash-consistent backup of the SQLite database
    using SQLite's native backup API (source_conn.backup(dest_conn)).
    Avoids corrupted snapshot risks associated with naive file-copying while writes are active.
    """
    source_path = db_path or DEFAULT_DB_PATH
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source database '{source_path}' does not exist.")

    if not target_backup_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_backup_path = f"{source_path}.backup_{timestamp}"

    backup_dir = os.path.dirname(os.path.abspath(target_backup_path))
    if backup_dir and not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)

    with open_db(source_path) as src_conn:
        dest_conn = sqlite3.connect(target_backup_path)
        try:
            src_conn.backup(dest_conn)
        finally:
            dest_conn.close()

    logger.info("Database backed up successfully to: %s", target_backup_path)
    return target_backup_path


# ==============================================================================
# USER MANAGEMENT & AUTHENTICATION HELPERS
# ==============================================================================

def create_user(
    username: str,
    password: str,
    full_name: str,
    role: str,
    db_path: Optional[str] = None,
    must_change_password: int = 0
) -> Dict[str, Any]:
    """
    Creates a new user account with a secure scrypt password hash.
    Enforces minimum length and allowed roles.
    """
    clean_user = (username or "").strip()
    if not clean_user or len(clean_user) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    clean_name = (full_name or "").strip()
    if not clean_name:
        raise ValueError("Full Name is required.")
    clean_role = (role or "").strip().lower()
    if clean_role not in ('admin', 'clinician', 'lab_staff'):
        raise ValueError("Role must be 'admin', 'clinician', or 'lab_staff'.")

    pw_hash = generate_password_hash(password, method="scrypt")
    target_path = db_path or DEFAULT_DB_PATH

    with open_db(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (clean_user,))
        if cursor.fetchone():
            raise ValueError(f"Username '{clean_user}' already exists.")

        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, role, is_active, must_change_password)
            VALUES (?, ?, ?, ?, 1, ?);
        """, (clean_user, pw_hash, clean_name, clean_role, must_change_password))
        conn.commit()
        new_id = cursor.lastrowid

    return get_user_by_id(new_id, db_path=target_path)


def get_user_by_id(user_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves user by ID (excludes password_hash from return dict for safety)."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, username, full_name, role, is_active, must_change_password, created_at, updated_at
            FROM users WHERE user_id = ?;
        """, (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_username(username: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves user by username (excludes password_hash from return dict for safety)."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, username, full_name, role, is_active, must_change_password, created_at, updated_at
            FROM users WHERE username = ?;
        """, ((username or "").strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def verify_user_credentials(
    username: str,
    password: str,
    db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Verifies username and password using constant-time hash check.
    Returns user dict (without password_hash) if credentials are valid and user is_active == 1.
    Returns None if invalid or deactivated.
    """
    clean_user = (username or "").strip()
    if not clean_user or not password:
        return None

    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, username, password_hash, full_name, role, is_active, must_change_password
            FROM users WHERE username = ?;
        """, (clean_user,))
        row = cursor.fetchone()
        if not row:
            return None

        # Check password hash
        if check_password_hash(row["password_hash"], password):
            if row["is_active"] != 1:
                return None  # Deactivated account
            return {
                "user_id": row["user_id"],
                "username": row["username"],
                "full_name": row["full_name"],
                "role": row["role"],
                "is_active": row["is_active"],
                "must_change_password": row["must_change_password"]
            }
        return None


def list_all_users(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists all users ordered by creation date (excludes password hashes)."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, username, full_name, role, is_active, must_change_password, created_at, updated_at
            FROM users ORDER BY created_at DESC;
        """)
        return [dict(r) for r in cursor.fetchall()]


def set_user_active_status(user_id: int, is_active: bool, db_path: Optional[str] = None) -> bool:
    """Toggles active/deactivated status of a user."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET is_active = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?;
        """, (1 if is_active else 0, user_id))
        conn.commit()
        return cursor.rowcount > 0


def reset_user_password(
    user_id: int,
    new_password: str,
    db_path: Optional[str] = None,
    must_change_password: bool = True
) -> bool:
    """Resets user password with a new scrypt hash."""
    if not new_password or len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    pw_hash = generate_password_hash(new_password, method="scrypt")
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users 
            SET password_hash = ?, 
                must_change_password = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?;
        """, (pw_hash, 1 if must_change_password else 0, user_id))
        conn.commit()
        return cursor.rowcount > 0


def change_user_password(
    user_id: int,
    old_password: str,
    new_password: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validates current password and updates to new password with scrypt hash.
    Clears must_change_password flag to 0.
    Enforces minimum length of 8 characters and prevents password reuse.
    """
    if not old_password:
        return False, "Current password is required."
    if not new_password or len(new_password) < 8:
        return False, "New password must be at least 8 characters."
    if old_password == new_password:
        return False, "New password cannot be the same as your current password."

    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, is_active FROM users WHERE user_id = ?;", (user_id,))
        row = cursor.fetchone()
        if not row:
            return False, "User not found."
        if row["is_active"] != 1:
            return False, "Account is deactivated."
        if not check_password_hash(row["password_hash"], old_password):
            return False, "Current password is incorrect."

        new_hash = generate_password_hash(new_password, method="scrypt")
        cursor.execute("""
            UPDATE users
            SET password_hash = ?,
                must_change_password = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?;
        """, (new_hash, user_id))
        conn.commit()
        return True, "Password changed successfully."


# ==============================================================================
# AUDIT LOGGING HELPERS
# ==============================================================================

def log_audit_event(
    action: Optional[str] = None,
    status: str = "SUCCESS",
    username: Optional[str] = None,
    user_id: Optional[int] = None,
    resource_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[str] = None,
    db_path: Optional[str] = None,
    **kwargs
) -> int:
    """
    Records an immutable audit event in SQLite.
    Never pass plaintext passwords or raw clinical biomarkers to details.
    """
    actual_action = action or kwargs.get("event_type") or "UNKNOWN_ACTION"
    target_path = db_path or DEFAULT_DB_PATH
    try:
        with open_db(target_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (user_id, username, action, resource_id, ip_address, status, details)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (user_id, username, actual_action, resource_id, ip_address, status, details))
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.warning("Failed to record audit log: %s", e)
        return -1


def get_recent_audit_logs(limit: int = 100, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves recent audit log events."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT log_id, user_id, username, action, resource_id, ip_address, status, details, created_at
            FROM audit_logs ORDER BY created_at DESC, log_id DESC LIMIT ?;
        """, (limit,))
        return [dict(r) for r in cursor.fetchall()]


# ==============================================================================
# RAG DOCUMENT GOVERNANCE (CHIEF DOCTOR AUTHORIZATION GATEKEEPER)
# ==============================================================================

def create_rag_document_submission(
    filename_or_data: Any,
    original_filename: Optional[str] = None,
    title: Optional[str] = None,
    file_path: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    file_hash_sha256: Optional[str] = None,
    uploaded_by_id: Optional[int] = None,
    uploaded_by_name: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """Inserts a new clinician document submission in 'pending_approval' status."""
    if isinstance(filename_or_data, dict):
        d = filename_or_data
        fn = d.get("filename")
        orig_fn = d.get("original_filename", fn)
        t = d.get("title", fn)
        fp = d.get("file_path")
        fsz = d.get("file_size_bytes", 0)
        fhash = d.get("file_hash_sha256")
        uid = d.get("uploaded_by_id")
        uname = d.get("uploaded_by_name")
        target_db = db_path or d.get("db_path")
    else:
        fn = filename_or_data
        orig_fn = original_filename
        t = title
        fp = file_path
        fsz = file_size_bytes
        fhash = file_hash_sha256
        uid = uploaded_by_id
        uname = uploaded_by_name
        target_db = db_path

    with open_db(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO rag_documents (
                filename, original_filename, title, file_path, file_size_bytes,
                file_hash_sha256, uploaded_by_id, uploaded_by_name, review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending_approval');
        """, (
            fn, orig_fn, t, fp, fsz,
            fhash, uid, uname
        ))
        conn.commit()
        return cursor.lastrowid


def get_rag_documents(
    status: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves RAG documents, optionally filtered by review_status ('pending_approval', 'approved', 'rejected')."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        if status:
            cursor.execute("""
                SELECT * FROM rag_documents WHERE review_status = ? ORDER BY created_at DESC;
            """, (status,))
        else:
            cursor.execute("""
                SELECT * FROM rag_documents ORDER BY created_at DESC;
            """)
        return [dict(r) for r in cursor.fetchall()]


def get_rag_document_by_id(doc_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single RAG document record by doc_id."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rag_documents WHERE doc_id = ?;", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_rag_document_status(
    doc_id: int,
    status: str,
    reviewed_by_id: Optional[int] = None,
    reviewed_by_name: Optional[str] = None,
    review_notes: Optional[str] = None,
    passages_count: Optional[int] = None,
    db_path: Optional[str] = None
) -> bool:
    """Updates review_status ('approved' or 'rejected') with Chief Doctor review metadata."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        if passages_count is not None:
            cursor.execute("""
                UPDATE rag_documents
                SET review_status = ?, reviewed_by_id = ?, reviewed_by_name = ?,
                    review_notes = ?, passages_count = ?, reviewed_at = CURRENT_TIMESTAMP
                WHERE doc_id = ?;
            """, (status, reviewed_by_id, reviewed_by_name, review_notes, passages_count, doc_id))
        else:
            cursor.execute("""
                UPDATE rag_documents
                SET review_status = ?, reviewed_by_id = ?, reviewed_by_name = ?,
                    review_notes = ?, reviewed_at = CURRENT_TIMESTAMP
                WHERE doc_id = ?;
            """, (status, reviewed_by_id, reviewed_by_name, review_notes, doc_id))
        conn.commit()
        return cursor.rowcount > 0


# ==============================================================================
# MULTI-SESSION PERSISTENT CHAT DATABASE LAYER
# ==============================================================================

def create_or_get_chat_session(
    session_id: str,
    user_id: int,
    username: str,
    title: str = "New Consultation",
    patient_id: Optional[str] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieves an existing chat session or creates a new persistent session."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?;", (session_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)

        cursor.execute("""
            INSERT INTO chat_sessions (session_id, user_id, username, title, patient_id)
            VALUES (?, ?, ?, ?, ?);
        """, (session_id, user_id, username, title, patient_id))
        conn.commit()
        cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?;", (session_id,))
        return dict(cursor.fetchone())


def get_user_chat_sessions(
    user_id: int,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves all chat sessions for a specific user, sorted newest first."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, 
                   (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.session_id) AS message_count
            FROM chat_sessions s
            WHERE s.user_id = ?
            ORDER BY s.updated_at DESC, s.created_at DESC;
        """, (user_id,))
        return [dict(r) for r in cursor.fetchall()]


def get_chat_session_by_id(session_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single chat session by session_id."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?;", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_chat_session_title(session_id: str, title: str, db_path: Optional[str] = None) -> bool:
    """Updates the title of a chat session."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat_sessions
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?;
        """, (title, session_id))
        conn.commit()
        return cursor.rowcount > 0


def add_chat_message(
    session_id: str,
    role: str,
    content: str,
    citations_json: Optional[str] = None,
    synthesis_engine: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """Appends a turn (user or assistant) to the persistent chat session and bumps updated_at."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages (session_id, role, content, citations_json, synthesis_engine)
            VALUES (?, ?, ?, ?, ?);
        """, (session_id, role, content, citations_json, synthesis_engine))
        cursor.execute("""
            UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?;
        """, (session_id,))
        conn.commit()
        return cursor.lastrowid


def get_chat_session_messages(
    session_id: str,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves all chronologically ordered messages in a chat session."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC, message_id ASC;
        """, (session_id,))
        return [dict(r) for r in cursor.fetchall()]


def delete_chat_session(
    session_id: str,
    user_id: Optional[int] = None,
    db_path: Optional[str] = None
) -> bool:
    """Deletes a chat session and cascades deletion of its messages."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute("DELETE FROM chat_sessions WHERE session_id = ? AND user_id = ?;", (session_id, user_id))
        else:
            cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?;", (session_id,))
        conn.commit()
        return cursor.rowcount > 0


# ==============================================================================
# RAG QUERY TELEMETRY & OBSERVABILITY
# ==============================================================================

def log_rag_query(
    query_text: str,
    response_text: str,
    synthesis_engine: str,
    citation_valid: int = 1,
    fallback_reason: Optional[str] = None,
    retrieved_passages_count: int = 0,
    sources_cited_json: Optional[str] = None,
    latency_ms: Optional[float] = None,
    user_id: Optional[int] = None,
    username: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """Records real-time RAG query telemetry for the observability dashboard."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO rag_query_logs (
                user_id, username, session_id, query_text, response_text,
                synthesis_engine, citation_valid, fallback_reason,
                retrieved_passages_count, sources_cited_json, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            user_id, username, session_id, query_text, response_text,
            synthesis_engine, citation_valid, fallback_reason,
            retrieved_passages_count, sources_cited_json, latency_ms
        ))
        conn.commit()
        return cursor.lastrowid


def get_rag_metrics_summary(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculates aggregate quality and performance metrics for the RAG monitor dashboard."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) AS total_queries,
                SUM(CASE WHEN citation_valid = 1 THEN 1 ELSE 0 END) AS valid_citations,
                SUM(CASE WHEN synthesis_engine = 'extractive' THEN 1 ELSE 0 END) AS fallback_count,
                SUM(CASE WHEN synthesis_engine = 'gemini' THEN 1 ELSE 0 END) AS gemini_count,
                AVG(latency_ms) AS avg_latency_ms
            FROM rag_query_logs;
        """)
        row = cursor.fetchone()
        total = row["total_queries"] or 0
        valid = row["valid_citations"] or 0
        fallbacks = row["fallback_count"] or 0
        gemini = row["gemini_count"] or 0
        avg_latency = round(row["avg_latency_ms"] or 0.0, 1)

        pass_rate = round((valid / total) * 100, 1) if total > 0 else 100.0
        fallback_rate = round((fallbacks / total) * 100, 1) if total > 0 else 0.0

        # Query guideline citations distribution
        cursor.execute("""
            SELECT sources_cited_json FROM rag_query_logs WHERE sources_cited_json IS NOT NULL;
        """)
        guideline_counts = {"AASLD": 0, "EASL-ERN": 0, "NIH": 0, "INASL/CPG": 0, "Other": 0}
        for log_row in cursor.fetchall():
            try:
                sources = json.loads(log_row[0]) if log_row[0] else []
                for s in sources:
                    src_str = s.get("source", "").lower() if isinstance(s, dict) else str(s).lower()
                    if "aasld" in src_str:
                        guideline_counts["AASLD"] += 1
                    elif "easl" in src_str:
                        guideline_counts["EASL-ERN"] += 1
                    elif "nih" in src_str:
                        guideline_counts["NIH"] += 1
                    elif "clinical practice" in src_str or "inasl" in src_str:
                        guideline_counts["INASL/CPG"] += 1
                    else:
                        guideline_counts["Other"] += 1
            except Exception:
                pass

        return {
            "total_queries": total,
            "valid_citations": valid,
            "pass_rate_pct": pass_rate,
            "fallback_count": fallbacks,
            "fallback_rate_pct": fallback_rate,
            "gemini_count": gemini,
            "avg_latency_ms": avg_latency,
            "guideline_counts": guideline_counts
        }


def get_recent_rag_queries(
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves paginated recent RAG queries with telemetry."""
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM rag_query_logs ORDER BY created_at DESC, log_id DESC LIMIT ? OFFSET ?;
        """, (limit, offset))
        return [dict(r) for r in cursor.fetchall()]


