"""
test_app_integration.py

Comprehensive Automated Integration Test Suite for NeuroMed AI Flask Application.
Validates:
1. Application startup and configurable database initialization.
2. API routes: GET /api/next_patient_id, GET /api/patient/<patient_id>, GET /patients.
3. Form submission -> Patient creation -> Assessment persistence -> Result rendering.
4. Reassessment adds historical records without overwriting baseline demographics.
5. Patient identity isolation: patient_id and patient_name never enter the 23 ML feature columns.
6. Validation rejection (HTTP 400) on malformed inputs without saving an assessment.
7. Prediction failure (HTTP 500) rollback without saving a partial assessment.
8. Medical RAG chat assistant route functionality.

Uses isolated temporary databases and synthetic clinical inputs.
"""

import os
import tempfile
import unittest
from unittest.mock import patch
import json
import pandas as pd

from app import app
from database import (
    init_db,
    open_db,
    get_patient_by_id,
    get_all_patients,
    get_patient_history
)


def get_synthetic_form_data(patient_name="Synthetic Test Patient", patient_id="NEUROMED001"):
    """Returns valid synthetic form parameters covering all 23 clinical features."""
    return {
        "patientName": patient_name,
        "patientId": patient_id,
        "inputAge": "32",
        "inputGender": "Female",
        "inputRegion": "North",
        "inputSocioeconomicStatus": "High",
        "inputBMI": "23.5",
        "inputAlcoholUse": "False",
        "inputCeruloplasmin": "26.0",
        "inputCopperBlood": "110.0",
        "inputFreeCopperBlood": "9.5",
        "inputCopperUrine": "28.0",
        "inputALT": "24.0",
        "inputAST": "28.0",
        "inputTotalBilirubin": "0.9",
        "inputAlbumin": "4.2",
        "inputALP": "88.0",
        "inputProthrombin": "1.05",
        "inputGGT": "32.0",
        "inputKFR": "No",
        "inputNeurological": "0.5",
        "inputPsychiatric": "No",
        "inputCognitive": "92.0",
        "inputFamilyHistory": "No",
        "inputGeneMutation": "No"
    }


class TestAppIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_app_records.db")
        init_db(self.db_path)

        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["DB_PATH"] = self.db_path
        self.client = app.test_client()

        # Provision and authenticate test clinician session
        from database import create_user
        import time
        user = create_user("test_clinician", "ClinicianPass123!", "Dr. Test Clinician", "clinician", db_path=self.db_path)
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

    def test_api_next_patient_id(self):
        """Verifies GET /api/next_patient_id returns initial sequential ID NEUROMED001."""
        response = self.client.get("/api/next_patient_id")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("next_patient_id"), "NEUROMED001")

    def test_api_patient_lookup(self):
        """Verifies GET /api/patient/<id> returns 404 when absent, 200 when present."""
        # Non-existent
        resp_404 = self.client.get("/api/patient/NEUROMED999")
        self.assertEqual(resp_404.status_code, 404)

        # Create patient via submit
        form_data = get_synthetic_form_data(patient_name="Arthur Dent", patient_id="NEUROMED001")
        submit_resp = self.client.post("/submit", data=form_data)
        self.assertEqual(submit_resp.status_code, 200)

        # Lookup created patient
        resp_200 = self.client.get("/api/patient/NEUROMED001")
        self.assertEqual(resp_200.status_code, 200)
        data = resp_200.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["patient"]["patient_name"], "Arthur Dent")
        self.assertEqual(data["patient"]["sex"], "Female")

    def test_patients_dashboard_route(self):
        """Verifies GET /patients renders directory view with stored patient summary."""
        # Submit test patient
        form_data = get_synthetic_form_data(patient_name="Ford Prefect", patient_id="NEUROMED001")
        self.client.post("/submit", data=form_data)

        # Access dashboard
        response = self.client.get("/patients")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Ford Prefect", html)
        self.assertIn("NEUROMED001", html)

    def test_patient_identity_isolation_from_model_features(self):
        """
        Verifies that patient identity (patientName, patientId) NEVER enters
        the 23-feature DataFrame passed into the ML prediction pipeline.
        """
        passed_dfs = []

        import app as app_module
        orig_safe_predict = app_module.safe_shap_and_predict

        def mock_safe_predict(df_input):
            passed_dfs.append(df_input.copy())
            return orig_safe_predict(df_input)

        with patch("app.safe_shap_and_predict", side_effect=mock_safe_predict):
            form_data = get_synthetic_form_data(patient_name="Secret Name", patient_id="NEUROMED001")
            response = self.client.post("/submit", data=form_data)
            self.assertEqual(response.status_code, 200)

        self.assertEqual(len(passed_dfs), 1)
        feature_df = passed_dfs[0]
        self.assertEqual(feature_df.shape[1], 23)
        self.assertNotIn("patientName", feature_df.columns)
        self.assertNotIn("patientId", feature_df.columns)
        self.assertNotIn("Secret Name", feature_df.values.flatten())
        self.assertNotIn("NEUROMED001", feature_df.values.flatten())

    def test_successful_submission_and_persistence(self):
        """Verifies form submission saves patient and assessment in the database."""
        form_data = get_synthetic_form_data(patient_name="Trillian Astra", patient_id="NEUROMED001")
        response = self.client.post("/submit", data=form_data)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        # Result page contains output
        self.assertIn("Predicted Risk", html)

        # Database verification
        patient = get_patient_by_id("NEUROMED001", self.db_path)
        self.assertIsNotNone(patient)
        self.assertEqual(patient["patient_name"], "Trillian Astra")

        history = get_patient_history("NEUROMED001", self.db_path)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["patient_id"], "NEUROMED001")
        self.assertIsNotNone(history[0]["probability"])
        self.assertIsNotNone(history[0]["leipzig_score"])

    def test_reassessment_adds_history_without_overwriting(self):
        """Verifies repeat assessment creates a second historical record without overwriting earlier visits."""
        form_data_1 = get_synthetic_form_data(patient_name="Zaphod Beeblebrox", patient_id="NEUROMED001")
        self.client.post("/submit", data=form_data_1)

        # Second assessment with changed urine copper
        form_data_2 = get_synthetic_form_data(patient_name="Zaphod Beeblebrox", patient_id="NEUROMED001")
        form_data_2["inputCopperUrine"] = "135.0" # High
        form_data_2["inputKFR"] = "Yes"
        self.client.post("/submit", data=form_data_2)

        history = get_patient_history("NEUROMED001", self.db_path)
        self.assertEqual(len(history), 2)
        # Newest first
        self.assertEqual(history[0]["copper_urine"], 135.0)
        self.assertEqual(history[0]["kayser_fleischer_rings"], 1)
        self.assertEqual(history[1]["copper_urine"], 28.0)
        self.assertEqual(history[1]["kayser_fleischer_rings"], 0)

    def test_invalid_input_rejection_prevents_persistence(self):
        """Verifies validation failure (HTTP 400) rejects malformed input and stores NO assessment."""
        form_data = get_synthetic_form_data(patient_name="Invalid Input Test", patient_id="NEUROMED001")
        # Set illegal cognitive score > 100
        form_data["inputCognitive"] = "150.0"

        response = self.client.post("/submit", data=form_data)
        self.assertEqual(response.status_code, 400)

        # Verify no assessment was saved
        with open_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assessments;")
            self.assertEqual(cursor.fetchone()[0], 0)

    def test_prediction_failure_rollback(self):
        """Verifies that an error during prediction pipeline rolls back and saves NO assessment."""
        with patch("app.safe_shap_and_predict", side_effect=RuntimeError("Simulated Model Failure")):
            form_data = get_synthetic_form_data(patient_name="Crash Test Patient", patient_id="NEUROMED001")
            response = self.client.post("/submit", data=form_data)
            self.assertEqual(response.status_code, 500)

        # Database must have 0 assessments
        with open_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assessments;")
            self.assertEqual(cursor.fetchone()[0], 0)

    def test_rag_chat_endpoint_remains_intact(self):
        """Verifies /api/chat medical chatbot returns strict message when medical_docs/ is empty."""
        chat_payload = {"message": "What is the normal reference range for serum ceruloplasmin?"}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(chat_payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()
        self.assertIn("response", res_data)
        self.assertTrue(
            res_data["response"] == "I do not have access to any medical documents and cannot answer." or
            "ceruloplasmin" in res_data["response"].lower()
        )

    def test_rag_chat_strictly_rejects_when_medical_docs_empty(self):
        """Verifies /api/chat strictly says it has no medical documents when medical_docs/ is empty."""
        chat_payload = {"message": "What does my 0.0% risk score mean?"}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(chat_payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()
        self.assertEqual(
            res_data["response"],
            "I do not have access to any medical documents and cannot answer."
        )


if __name__ == "__main__":
    unittest.main()
