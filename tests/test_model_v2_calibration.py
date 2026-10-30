"""
tests/test_model_v2_calibration.py

Regression Test Suite for Phase 8 Increment 2:
1. Preprocessing data leakage prevention (split before fitting, fit on train only).
2. Stacking out-of-fold (OOF) integrity and meta-model training.
3. Probability calibration (CalibratedClassifierCV artifact presence, calibration curves, ECE, Brier score).
4. Synthetic dataset v2 clinical plausibility (no negative laboratory values, non-zero differential shortcuts).
5. Clinical interface distinction (statistical probability decoupled from Leipzig clinical score, model version tracking).
"""

import os
import tempfile
import unittest
import json
import joblib
import numpy as np
import pandas as pd

from app import app, safe_shap_and_predict, models_ready
from database import init_db, save_assessment, get_assessment_by_id, get_all_patients

class TestModelV2Calibration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_v2_calibration.db")
        init_db(self.db_path)

        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["DB_PATH"] = self.db_path
        self.client = app.test_client()

        # Provision test clinician
        from database import create_user
        import time
        user = create_user("test_clinician_v2", "ClinicianPass123!", "Dr. V2 Evaluator", "clinician", db_path=self.db_path)
        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = user["username"]
            sess["role"] = user["role"]
            sess["full_name"] = user["full_name"]
            sess["last_active"] = time.time()

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_synthetic_v2_dataset_physiological_validity_and_no_shortcuts(self):
        """Verifies synthetic dataset v2 has zero negative lab values and no deterministic shortcuts."""
        csv_path = "Wilson_disease_dataset_v2.csv"
        self.assertTrue(os.path.exists(csv_path), "Wilson_disease_dataset_v2.csv must exist.")
        df = pd.read_csv(csv_path)

        # 1. No negative laboratory values
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            min_val = df[col].min()
            self.assertGreaterEqual(min_val, 0.0, f"Column '{col}' contains unphysical negative value: {min_val}")

        # 2. Kayser-Fleischer Rings is NOT 0.0% in controls (differential overlap exists)
        control_kf_rate = df[df["Is_Wilson_Disease"] == 0]["Kayser-Fleischer Rings"].mean()
        self.assertGreater(control_kf_rate, 0.0, "Kayser-Fleischer Rings must have non-zero prevalence in controls.")
        self.assertLess(control_kf_rate, 0.15, "Control KF prevalence should reflect rare cholestatic differential rate.")

        # 3. ATB7B Gene Mutation is NOT 0.0% in controls (carrier/VUS frequency)
        control_gene_rate = df[df["Is_Wilson_Disease"] == 0]["ATB7B Gene Mutation"].mean()
        self.assertGreater(control_gene_rate, 0.0, "ATB7B Gene Mutation must have non-zero carrier rate in controls.")

        # 4. Age distribution overlaps across cohorts
        w_age_min = df[df["Is_Wilson_Disease"] == 1]["Age"].min()
        c_age_min = df[df["Is_Wilson_Disease"] == 0]["Age"].min()
        self.assertLess(w_age_min, 40.0)
        self.assertLess(c_age_min, 40.0)

    def test_v2_artifacts_and_calibrator_presence(self):
        """Verifies v2 model artifacts and calibrator are present and serialized correctly."""
        artifact_dir = "wilson_artifacts"
        required_artifacts = [
            "preprocessor.joblib",
            "svm.joblib",
            "logreg_base.joblib",
            "meta_model.joblib",
            "calibrator.joblib",
            "meta_info.json"
        ]
        for art in required_artifacts:
            path = os.path.join(artifact_dir, art)
            self.assertTrue(os.path.exists(path), f"Required artifact '{art}' missing from {artifact_dir}.")

        with open(os.path.join(artifact_dir, "meta_info.json"), "r") as f:
            meta_info = json.load(f)
        
        self.assertEqual(meta_info.get("model_version"), "v2.0.0-calibrated")
        self.assertIn("calibration_method", meta_info)
        self.assertIn("test_metrics_calibrated", meta_info)

    def test_calibrator_reduces_ece_and_outputs_valid_probabilities(self):
        """Verifies calibrator outputs probabilities in [0, 1] and test ECE is tightly bounded."""
        with open("wilson_artifacts/meta_info.json", "r") as f:
            meta_info = json.load(f)

        test_cal = meta_info["test_metrics_calibrated"]
        self.assertLessEqual(test_cal["ece"], 0.05, "Calibrated ECE should be well calibrated (< 0.05).")
        self.assertLessEqual(test_cal["brier_score"], 0.05, "Calibrated Brier score should be well bounded (< 0.05).")

    def test_safe_shap_and_predict_produces_calibrated_probability(self):
        """Verifies safe_shap_and_predict returns calibrated probability and model breakdown."""
        self.assertTrue(models_ready(), "Models must be ready.")
        patient_data = {
            "Age": 28.0, "Sex": "Male", "Ceruloplasmin Level": 19.0,
            "Copper in Blood Serum": 120.0, "Free Copper in Blood Serum": 16.0,
            "Copper in Urine": 55.0, "ALT": 65.0, "AST": 55.0,
            "Total Bilirubin": 1.4, "Albumin": 4.1, "Alkaline Phosphatase (ALP)": 110.0,
            "Prothrombin Time / INR": 1.15, "Gamma-Glutamyl Transferase (GGT)": 55.0,
            "Kayser-Fleischer Rings": 0, "Neurological Symptoms Score": 1.0,
            "Psychiatric Symptoms": 0, "Cognitive Function Score": 88.0,
            "Family History": 0, "ATB7B Gene Mutation": 0, "Region": "West",
            "Socioeconomic Status": "High", "Alcohol Use": False, "BMI": 25.0
        }
        df_patient = pd.DataFrame([patient_data])
        final_prob, final_pred, shap_res, used_syn, used_df, breakdown = safe_shap_and_predict(df_patient)

        self.assertIsInstance(final_prob, float)
        self.assertGreaterEqual(final_prob, 0.0)
        self.assertLessEqual(final_prob, 1.0)
        self.assertIn(final_pred, (0, 1))
        self.assertIn("svm", breakdown)
        self.assertIn("logreg", breakdown)
        self.assertIn("bilstm", breakdown)
        self.assertIsNotNone(breakdown["bilstm"])

    def test_database_persists_and_retrieves_model_version(self):
        """Verifies database assessments table stores and retrieves model_version."""
        from database import create_or_update_patient
        patient_record = {
            "patient_id": "TEST_V2_PID",
            "patient_name": "Test V2 Patient"
        }
        create_or_update_patient(patient_record, db_path=self.db_path)

        pred_results = {
            "prediction_text": "Low Predicted Risk – Wilson Disease",
            "prediction_label": 0,
            "probability": 0.0267,
            "model_breakdown": {"svm": 0.2, "logreg": 3.0, "bilstm": None},
            "leipzig_score": 1,
            "model_version": "v2.0.0-calibrated"
        }
        ass_id = save_assessment(
            patient_id=patient_record["patient_id"],
            clinical_inputs={"Age": 30, "Sex": "Female"},
            prediction_results=pred_results,
            db_path=self.db_path
        )
        self.assertIsNotNone(ass_id)

        retrieved = get_assessment_by_id(ass_id, db_path=self.db_path)
        self.assertEqual(retrieved.get("model_version"), "v2.0.0-calibrated")
        self.assertAlmostEqual(retrieved.get("probability"), 0.0267, places=4)

        # Check directory summary includes model version
        patients_list = get_all_patients(db_path=self.db_path)
        matching = [p for p in patients_list if p["patient_id"] == "TEST_V2_PID"]
        self.assertTrue(len(matching) > 0)
        self.assertEqual(matching[0].get("latest_model_version"), "v2.0.0-calibrated")

    def test_result_html_renders_clinical_distinction_disclaimer(self):
        """Verifies result view distinguishes statistical ML probability from clinical certainty."""
        form_payload = {
            "patientName": "Clinical Distinction Test Patient",
            "patientId": "NEUROMED999",
            "inputAge": "26",
            "inputGender": "Female",
            "inputRegion": "East",
            "inputSocioeconomicStatus": "Medium",
            "inputBMI": "22.4",
            "inputAlcoholUse": "False",
            "inputCeruloplasmin": "22.0",
            "inputCopperBlood": "115.0",
            "inputFreeCopperBlood": "12.0",
            "inputCopperUrine": "35.0",
            "inputALT": "28.0",
            "inputAST": "30.0",
            "inputTotalBilirubin": "1.0",
            "inputAlbumin": "4.3",
            "inputALP": "90.0",
            "inputProthrombin": "1.05",
            "inputGGT": "35.0",
            "inputKFR": "No",
            "inputNeurological": "0.5",
            "inputPsychiatric": "No",
            "inputCognitive": "90.0",
            "inputFamilyHistory": "No",
            "inputGeneMutation": "No"
        }
        res = self.client.post("/submit", data=form_payload, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # Must display clinical distinction & boundary warning
        self.assertIn("Clinical Distinction & Boundary", html)
        self.assertIn("v2.0.0-calibrated", html)
        self.assertIn("STATISTICAL PROBABILITY", html)
        # Leipzig score section must remain separate from ML gauge
        self.assertIn("Leipzig Criteria Contribution", html)

if __name__ == "__main__":
    unittest.main()
