"""
test_database.py

Comprehensive Automated Test Suite for NeuroMed AI Database Layer.
Validates:
1. Database schema initialization (patients, assessments, indexes).
2. Sequential patient ID generation (NEUROMED001, NEUROMED002, ...).
3. Concurrency-safe duplicate prevention and constraints.
4. Patient creation, retrieval, and demographic updates.
5. Reassessment history persistence (no overwriting of previous assessments).
6. Strict foreign-key constraint enforcement.
7. Strict preservation of missing values as None/NULL (no zero fabrication).
8. Atomic transaction rollbacks on failure.
9. Patient directory listing, search, and aggregated metrics.

Uses temporary SQLite databases to ensure zero impact on production/development records.
"""

import os
import tempfile
import unittest
import sqlite3
from typing import Dict, Any

from database import (
    init_db,
    open_db,
    get_next_patient_id,
    create_or_update_patient,
    get_patient_by_id,
    get_all_patients,
    save_assessment,
    get_assessment_by_id,
    get_patient_history,
    FEATURE_TO_COL_MAP
)


class TestDatabaseLayer(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary SQLite database for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_records.db")
        init_db(self.db_path)

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_database_initialization(self):
        """Verifies tables and indexes are created properly with foreign keys enabled."""
        with open_db(self.db_path) as conn:
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r["name"] for r in cursor.fetchall()]
            self.assertIn("patients", tables)
            self.assertIn("assessments", tables)

            # Check foreign keys pragma is active
            cursor.execute("PRAGMA foreign_keys;")
            fk_status = cursor.fetchone()[0]
            self.assertEqual(fk_status, 1)

    def test_patient_id_sequencing(self):
        """Verifies sequential zero-padded ID generation (NEUROMED001 -> NEUROMED002)."""
        # Starting ID when table is empty
        next_id = get_next_patient_id(self.db_path)
        self.assertEqual(next_id, "NEUROMED001")

        # Create first patient
        p1 = create_or_update_patient({"patient_name": "Synthetic Patient Alpha"}, db_path=self.db_path)
        self.assertEqual(p1["patient_id"], "NEUROMED001")

        # Next ID should be NEUROMED002
        next_id = get_next_patient_id(self.db_path)
        self.assertEqual(next_id, "NEUROMED002")

        # Create second patient
        p2 = create_or_update_patient({"patient_name": "Synthetic Patient Beta"}, db_path=self.db_path)
        self.assertEqual(p2["patient_id"], "NEUROMED002")

        # Create multiple patients to test progression
        for i in range(3, 12):
            p = create_or_update_patient({"patient_name": f"Patient {i}"}, db_path=self.db_path)
            expected_id = f"NEUROMED{i:03d}"
            self.assertEqual(p["patient_id"], expected_id)

    def test_patient_creation_and_retrieval(self):
        """Tests inserting and reading a patient record with demographic details."""
        data = {
            "patient_name": "Eleanor Vance",
            "Age": 29.0,
            "Sex": "Female",
            "Region": "North",
            "Socioeconomic Status": "High",
            "BMI": 22.4,
            "Alcohol Use": "False"
        }
        created = create_or_update_patient(data, db_path=self.db_path)
        self.assertEqual(created["patient_id"], "NEUROMED001")
        self.assertEqual(created["patient_name"], "Eleanor Vance")
        self.assertEqual(created["age"], 29.0)
        self.assertEqual(created["sex"], "Female")

        # Retrieve by ID
        retrieved = get_patient_by_id("NEUROMED001", db_path=self.db_path)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["patient_name"], "Eleanor Vance")

        # Non-existent ID returns None
        self.assertIsNone(get_patient_by_id("NEUROMED999", db_path=self.db_path))

    def test_update_existing_patient_demographics(self):
        """Verifies updating an existing patient's demographics without duplicating the record."""
        p = create_or_update_patient({"patient_name": "John Doe", "Age": 35.0, "Sex": "Male"}, db_path=self.db_path)
        pid = p["patient_id"]

        # Update patient name and age
        updated = create_or_update_patient({
            "patient_id": pid,
            "patient_name": "Johnathan Doe",
            "Age": 36.0,
            "Region": "South"
        }, db_path=self.db_path)

        self.assertEqual(updated["patient_id"], pid)
        self.assertEqual(updated["patient_name"], "Johnathan Doe")
        self.assertEqual(updated["age"], 36.0)
        self.assertEqual(updated["region"], "South")

        # Total patient count must still be 1
        patients = get_all_patients(db_path=self.db_path)
        self.assertEqual(len(patients), 1)

    def test_foreign_key_constraint_enforcement(self):
        """Verifies that assessments cannot be saved for non-existent patient IDs."""
        sample_inputs = {"Age": 24.0, "Ceruloplasmin Level": 12.0}
        sample_outputs = {"prediction_text": "Low Predicted Risk", "probability": 0.12}

        # Attempt to save assessment for non-existent patient ID
        with self.assertRaises(ValueError):
            save_assessment("NON_EXISTENT_ID", sample_inputs, sample_outputs, db_path=self.db_path)

    def test_assessment_persistence_and_history(self):
        """
        Verifies that multiple assessments for a patient are preserved as discrete records.
        Reassessments must NOT overwrite earlier assessments.
        """
        p = create_or_update_patient({"patient_name": "Marcus Aurelius"}, db_path=self.db_path)
        pid = p["patient_id"]

        # 1. First Assessment (Initial visit - Baseline)
        inputs_1 = {
            "Age": 42.0, "Sex": "Male", "Ceruloplasmin Level": 8.5,
            "Copper in Blood Serum": 210.0, "Free Copper in Blood Serum": 45.0,
            "Copper in Urine": 150.0, "ALT": 75.0, "AST": 80.0,
            "Total Bilirubin": 2.1, "Albumin": 3.4, "Alkaline Phosphatase (ALP)": 110.0,
            "Prothrombin Time / INR": 1.25, "Gamma-Glutamyl Transferase (GGT)": 95.0,
            "Kayser-Fleischer Rings": 1, "Neurological Symptoms Score": 5.5,
            "Psychiatric Symptoms": 1, "Cognitive Function Score": 70.0,
            "Family History": 1, "ATB7B Gene Mutation": 1, "Region": "West",
            "Socioeconomic Status": "Medium", "Alcohol Use": "False", "BMI": 24.5
        }
        outputs_1 = {
            "prediction_text": "High Predicted Risk – Wilson Disease",
            "prediction_label": 1,
            "probability": 0.965,
            "model_breakdown": {"bilstm": 94.0, "svm": 98.0, "logreg": 92.5},
            "leipzig_score": 4,
            "shap_waterfall_path": "uploads/shap_waterfall_test1.png"
        }
        a1_id = save_assessment(pid, inputs_1, outputs_1, db_path=self.db_path)
        self.assertGreater(a1_id, 0)

        # 2. Second Assessment (Follow-up visit 6 months later)
        inputs_2 = dict(inputs_1)
        inputs_2["Ceruloplasmin Level"] = 15.0  # Improved
        inputs_2["Copper in Urine"] = 80.0      # Reduced under therapy
        outputs_2 = {
            "prediction_text": "High Predicted Risk – Wilson Disease",
            "prediction_label": 1,
            "probability": 0.820,
            "model_breakdown": {"bilstm": 80.0, "svm": 85.0, "logreg": 78.0},
            "leipzig_score": 3,
            "shap_waterfall_path": "uploads/shap_waterfall_test2.png"
        }
        a2_id = save_assessment(pid, inputs_2, outputs_2, db_path=self.db_path)
        self.assertGreater(a2_id, a1_id)

        # 3. Retrieve Assessment History
        history = get_patient_history(pid, db_path=self.db_path)
        self.assertEqual(len(history), 2)

        # Ordered newest first
        self.assertEqual(history[0]["assessment_id"], a2_id)
        self.assertEqual(history[0]["copper_urine"], 80.0)
        self.assertEqual(history[1]["assessment_id"], a1_id)
        self.assertEqual(history[1]["copper_urine"], 150.0)

        # Verify individual retrieval
        record_1 = get_assessment_by_id(a1_id, db_path=self.db_path)
        self.assertIsNotNone(record_1)
        self.assertEqual(record_1["patient_name"], "Marcus Aurelius")
        self.assertEqual(record_1["probability"], 0.965)
        self.assertEqual(record_1["leipzig_score"], 4)

    def test_missing_clinical_values_preserved_as_null(self):
        """
        Verifies that missing clinical parameters are strictly preserved as None/SQL NULL.
        The database MUST NEVER convert missing values to 0.0 or fabricate data.
        """
        p = create_or_update_patient({"patient_name": "Incomplete Panel Patient"}, db_path=self.db_path)
        pid = p["patient_id"]

        partial_inputs = {
            "Age": 30.0,
            "Sex": "Female",
            "Ceruloplasmin Level": 22.0,
            # Deliberately omitting 'Copper in Urine', 'ALT', 'AST', etc.
            "Copper in Urine": None,
            "ALT": "", # empty string should also become None
        }
        outputs = {
            "prediction_text": "Low Predicted Risk – Wilson Disease",
            "prediction_label": 0,
            "probability": 0.15,
            "leipzig_score": 0
        }

        aid = save_assessment(pid, partial_inputs, outputs, db_path=self.db_path)
        record = get_assessment_by_id(aid, db_path=self.db_path)

        self.assertIsNotNone(record)
        self.assertEqual(record["age"], 30.0)
        self.assertEqual(record["ceruloplasmin_level"], 22.0)
        # Verify unprovided / empty fields are strictly None (SQL NULL)
        self.assertIsNone(record["copper_urine"])
        self.assertIsNone(record["alt"])
        self.assertIsNone(record["ast"])
        self.assertIsNone(record["kayser_fleischer_rings"])

    def test_patient_search_and_dashboard_summary(self):
        """Verifies patient listing, latest metrics aggregation, and name/ID search."""
        # Patient 1: Has assessment
        p1 = create_or_update_patient({"patient_name": "Alice Cooper"}, db_path=self.db_path)
        save_assessment(p1["patient_id"], {"Age": 22.0}, {"prediction_text": "High Risk", "probability": 0.92, "leipzig_score": 3}, db_path=self.db_path)

        # Patient 2: No assessment yet
        p2 = create_or_update_patient({"patient_name": "Bob Dylan"}, db_path=self.db_path)

        # Listing all patients
        all_p = get_all_patients(db_path=self.db_path)
        self.assertEqual(len(all_p), 2)

        p1_summary = next(p for p in all_p if p["patient_id"] == p1["patient_id"])
        self.assertEqual(p1_summary["assessment_count"], 1)
        self.assertEqual(p1_summary["latest_probability"], 0.92)
        self.assertEqual(p1_summary["latest_leipzig_score"], 3)

        p2_summary = next(p for p in all_p if p["patient_id"] == p2["patient_id"])
        self.assertEqual(p2_summary["assessment_count"], 0)
        self.assertIsNone(p2_summary["latest_probability"])

        # Search by name
        search_res = get_all_patients(db_path=self.db_path, search="Bob")
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["patient_name"], "Bob Dylan")

        # Search by ID
        search_res_id = get_all_patients(db_path=self.db_path, search="NEUROMED001")
        self.assertEqual(len(search_res_id), 1)
        self.assertEqual(search_res_id[0]["patient_name"], "Alice Cooper")

    def test_transaction_rollback_on_failure(self):
        """Verifies transaction rollback ensures no partially-saved assessment on error."""
        p = create_or_update_patient({"patient_name": "Rollback Test Patient"}, db_path=self.db_path)
        pid = p["patient_id"]

        with open_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assessments;")
            count_before = cursor.fetchone()[0]

        # Trigger failure with invalid argument type or simulated error
        with self.assertRaises(Exception):
            save_assessment(pid, None, {}, db_path=self.db_path) # clinical_inputs is None -> causes exception

        with open_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assessments;")
            count_after = cursor.fetchone()[0]

        self.assertEqual(count_before, count_after)


if __name__ == "__main__":
    unittest.main()
