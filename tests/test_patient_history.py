"""
tests/test_patient_history.py

Automated test suite for Stage 5:
- Patient directory dashboard (zero, one, multiple patients)
- Patient search, risk filtering, and sorting
- Longitudinal assessment history ordering (newest first)
- Historical assessment report retrieval by ID
- Verification that historical report viewing does NOT invoke ML models
- Handling of missing patients and missing assessment IDs (404)
- Missing SHAP files and path traversal defense
- Patient data escaping in templates
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch

from app import app
from database import (
    init_db,
    create_or_update_patient,
    save_assessment,
    get_patient_by_id,
    open_db
)


class TestPatientDirectoryAndHistory(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_history_records.db")
        init_db(self.test_db)
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["DB_PATH"] = self.test_db
        self.client = app.test_client()

        from database import create_user
        import time
        user = create_user("test_clinician", "ClinicianPass123!", "Dr. Test Clinician", "clinician", db_path=self.test_db)
        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = user["username"]
            sess["role"] = user["role"]
            sess["full_name"] = user["full_name"]
            sess["last_active"] = time.time()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_clinical_inputs(self, **overrides):
        base = {
            "Age": 28.0,
            "Sex": "Female",
            "Ceruloplasmin Level": 25.0,
            "Copper in Blood Serum": 110.0,
            "Free Copper in Blood Serum": 9.5,
            "Copper in Urine": 28.0,
            "ALT": 25.0,
            "AST": 28.0,
            "Total Bilirubin": 0.9,
            "Albumin": 4.2,
            "Alkaline Phosphatase (ALP)": 90.0,
            "Prothrombin Time / INR": 1.05,
            "Gamma-Glutamyl Transferase (GGT)": 30.0,
            "Kayser-Fleischer Rings": 0,
            "Neurological Symptoms Score": 0.5,
            "Psychiatric Symptoms": 0,
            "Cognitive Function Score": 92.0,
            "Family History": 0,
            "ATB7B Gene Mutation": 0,
            "Region": "North",
            "Socioeconomic Status": "High",
            "Alcohol Use": "False",
            "BMI": 22.4
        }
        base.update(overrides)
        return base

    def test_dashboard_zero_patients(self):
        """Verifies dashboard displays clean empty state when 0 patients exist."""
        res = self.client.get("/patients")
        self.assertEqual(res.status_code, 200)
        content = res.get_data(as_text=True)
        self.assertIn("No Matching Patient Records Found", content)
        self.assertIn("No patient assessments have been saved yet", content)

    def test_dashboard_one_and_multiple_patients(self):
        """Verifies table renders correctly with multiple patients."""
        p1 = create_or_update_patient({"patient_id": "NEUROMED001", "patient_name": "Alice Cooper"}, db_path=self.test_db)
        p2 = create_or_update_patient({"patient_id": "NEUROMED002", "patient_name": "Bob Dylan"}, db_path=self.test_db)

        # Save an assessment for Bob
        save_assessment(
            patient_id="NEUROMED002",
            clinical_inputs=self._create_sample_clinical_inputs(),
            prediction_results={
                "prediction_text": "High Predicted Risk – Wilson Disease",
                "prediction_label": 1,
                "probability": 0.885,
                "leipzig_score": 4
            },
            db_path=self.test_db
        )

        res = self.client.get("/patients")
        self.assertEqual(res.status_code, 200)
        content = res.get_data(as_text=True)
        self.assertIn("NEUROMED001", content)
        self.assertIn("Alice Cooper", content)
        self.assertIn("NEUROMED002", content)
        self.assertIn("Bob Dylan", content)
        self.assertIn("88.5%", content)
        self.assertIn("High Risk", content)

    def test_dashboard_search_and_risk_filters(self):
        """Tests patient search and risk filtering on the dashboard."""
        p_high = create_or_update_patient({"patient_id": "NEUROMED001", "patient_name": "High Risk Patient"}, db_path=self.test_db)
        p_low = create_or_update_patient({"patient_id": "NEUROMED002", "patient_name": "Low Risk Patient"}, db_path=self.test_db)
        p_un = create_or_update_patient({"patient_id": "NEUROMED003", "patient_name": "Unassessed Patient"}, db_path=self.test_db)

        save_assessment(
            patient_id="NEUROMED001",
            clinical_inputs=self._create_sample_clinical_inputs(),
            prediction_results={"prediction_text": "High Risk", "prediction_label": 1, "probability": 0.92, "leipzig_score": 4},
            db_path=self.test_db
        )
        save_assessment(
            patient_id="NEUROMED002",
            clinical_inputs=self._create_sample_clinical_inputs(),
            prediction_results={"prediction_text": "Low Risk", "prediction_label": 0, "probability": 0.12, "leipzig_score": 0},
            db_path=self.test_db
        )

        # 1. Search by name
        res_search = self.client.get("/patients?search=Low")
        content_search = res_search.get_data(as_text=True)
        self.assertIn("Low Risk Patient", content_search)
        self.assertNotIn("High Risk Patient", content_search)

        # 2. Filter by risk = high
        res_high = self.client.get("/patients?risk=high")
        content_high = res_high.get_data(as_text=True)
        self.assertIn("High Risk Patient", content_high)
        self.assertNotIn("Low Risk Patient", content_high)
        self.assertNotIn("Unassessed Patient", content_high)

        # 3. Filter by risk = unassessed
        res_un = self.client.get("/patients?risk=unassessed")
        content_un = res_un.get_data(as_text=True)
        self.assertIn("Unassessed Patient", content_un)
        self.assertNotIn("High Risk Patient", content_un)

    def test_patient_history_ordering(self):
        """Verifies multiple assessments appear in newest-first chronological order."""
        p = create_or_update_patient({"patient_id": "NEUROMED001", "patient_name": "Longitudinal Patient"}, db_path=self.test_db)

        a1 = save_assessment(
            patient_id="NEUROMED001",
            clinical_inputs=self._create_sample_clinical_inputs(Age=20),
            prediction_results={"prediction_text": "Initial Assessment", "probability": 0.35, "leipzig_score": 1},
            db_path=self.test_db
        )
        a2 = save_assessment(
            patient_id="NEUROMED001",
            clinical_inputs=self._create_sample_clinical_inputs(Age=21),
            prediction_results={"prediction_text": "Follow-Up Assessment", "probability": 0.85, "leipzig_score": 4},
            db_path=self.test_db
        )

        res = self.client.get("/patient/NEUROMED001/history")
        self.assertEqual(res.status_code, 200)
        content = res.get_data(as_text=True)
        self.assertIn("Longitudinal Patient", content)
        self.assertIn("NEUROMED001", content)
        self.assertIn(f"#{a1}", content)
        self.assertIn(f"#{a2}", content)

        # Verify a2 (newest) appears before a1 (older) in the table content
        target_a2 = f"class=\"badge-id\">#{a2}</span>"
        target_a1 = f"class=\"badge-id\">#{a1}</span>"
        idx_a2 = content.find(target_a2)
        idx_a1 = content.find(target_a1)
        self.assertNotEqual(idx_a2, -1)
        self.assertNotEqual(idx_a1, -1)
        self.assertTrue(idx_a2 < idx_a1, f"Expected assessment #{a2} to appear before #{a1}")

    def test_historical_report_retrieval(self):
        """Tests viewing a specific historical assessment report."""
        p = create_or_update_patient({"patient_id": "NEUROMED001", "patient_name": "Eleanor Vance"}, db_path=self.test_db)
        a_id = save_assessment(
            patient_id="NEUROMED001",
            clinical_inputs=self._create_sample_clinical_inputs(Age=24.0, Ceruloplasmin_Level=12.5),
            prediction_results={
                "prediction_text": "High Predicted Risk – Wilson Disease",
                "prediction_label": 1,
                "probability": 0.8245,
                "leipzig_score": 3,
                "model_breakdown": {"svm": 85.0, "logreg": 80.0, "bilstm": None}
            },
            db_path=self.test_db
        )

        res = self.client.get(f"/assessment/{a_id}/report")
        self.assertEqual(res.status_code, 200)
        content = res.get_data(as_text=True)
        self.assertIn("Eleanor Vance", content)
        self.assertIn("NEUROMED001", content)
        self.assertIn(f"Archived Assessment #{a_id}", content)
        self.assertIn("82.5%", content)
        self.assertIn("High Predicted Risk", content)

    def test_historical_report_no_ml_execution(self):
        """Verifies that viewing an archived historical report does NOT invoke ML models."""
        p = create_or_update_patient({"patient_id": "NEUROMED001", "patient_name": "Static Patient"}, db_path=self.test_db)
        a_id = save_assessment(
            patient_id="NEUROMED001",
            clinical_inputs=self._create_sample_clinical_inputs(),
            prediction_results={"prediction_text": "Low Risk", "probability": 0.15, "leipzig_score": 0},
            db_path=self.test_db
        )

        # Patch safe_shap_and_predict to verify it is NOT called
        with patch("app.safe_shap_and_predict") as mock_ml:
            res = self.client.get(f"/assessment/{a_id}/report")
            self.assertEqual(res.status_code, 200)
            mock_ml.assert_not_called()

    def test_missing_patient_and_assessment_handling(self):
        """Verifies proper 404 responses for non-existent patients and assessments."""
        res_p = self.client.get("/patient/NON_EXISTENT_ID/history")
        self.assertEqual(res_p.status_code, 404)
        self.assertIn("not found", res_p.get_data(as_text=True).lower())

        res_a = self.client.get("/assessment/999999/report")
        self.assertEqual(res_a.status_code, 404)
        self.assertIn("not found", res_a.get_data(as_text=True).lower())

    def test_missing_shap_files_and_path_traversal(self):
        """Verifies that path traversal attacks in SHAP path are rejected and missing files render cleanly."""
        p = create_or_update_patient({"patient_id": "NEUROMED001", "patient_name": "Security Test Patient"}, db_path=self.test_db)
        
        # 1. Path traversal attempt
        a_id = save_assessment(
            patient_id="NEUROMED001",
            clinical_inputs=self._create_sample_clinical_inputs(),
            prediction_results={
                "prediction_text": "Low Risk",
                "probability": 0.20,
                "shap_waterfall_path": "../../windows/system32/cmd.exe"
            },
            db_path=self.test_db
        )

        res = self.client.get(f"/assessment/{a_id}/report")
        self.assertEqual(res.status_code, 200)
        content = res.get_data(as_text=True)
        # Traversal file must NOT be served or referenced
        self.assertNotIn("cmd.exe", content)
        self.assertIn("SHAP feature attribution waterfall plot was not recorded or is no longer available", content)

    def test_patient_data_escaping(self):
        """Verifies XSS attempts in patient name are escaped in all views."""
        xss_name = "<script>alert('xss')</script>"
        p = create_or_update_patient({"patient_id": "NEUROMED001", "patient_name": xss_name}, db_path=self.test_db)
        a_id = save_assessment(
            patient_id="NEUROMED001",
            clinical_inputs=self._create_sample_clinical_inputs(),
            prediction_results={"prediction_text": "Low Risk", "probability": 0.1},
            db_path=self.test_db
        )

        # 1. Directory
        res1 = self.client.get("/patients")
        self.assertNotIn("<script>alert('xss')</script>", res1.get_data(as_text=True))

        # 2. History
        res2 = self.client.get("/patient/NEUROMED001/history")
        self.assertNotIn("<script>alert('xss')</script>", res2.get_data(as_text=True))

        # 3. Report
        res3 = self.client.get(f"/assessment/{a_id}/report")
        self.assertNotIn("<script>alert('xss')</script>", res3.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
