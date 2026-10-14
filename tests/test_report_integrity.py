"""
tests/test_report_integrity.py

Automated test suite for Stage 6:
- Historical report retrieval does NOT invoke ML models or SHAP explainers
- Stored prediction, probability, inputs, and timestamp match the exact requested assessment
- Assessment retrieval does not leak or default to the patient's latest assessment
- Missing historical fields (null inputs, missing model breakdowns) are handled safely
- Missing or deleted SHAP artifact files render clean fallbacks without 500 error
- Print/PDF styling rules: hide navigation, interactive toolbars, chat widgets
- Verify accurate button label: "Print Report / Save as PDF"
- Verify prominent Clinical Decision Support disclaimer and recalculation notices
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
    open_db
)


class TestReportIntegrityAndPrintReview(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_integrity_records.db")
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

    def _sample_inputs(self, overrides_dict=None, **kwargs):
        data = {
            "Age": 32.0,
            "Sex": "Male",
            "Ceruloplasmin Level": 14.5,
            "Copper in Blood Serum": 135.0,
            "Free Copper in Blood Serum": 18.2,
            "Copper in Urine": 120.0,
            "ALT": 72.0,
            "AST": 68.0,
            "Total Bilirubin": 2.1,
            "Albumin": 3.8,
            "Alkaline Phosphatase (ALP)": 145.0,
            "Prothrombin Time / INR": 1.25,
            "Gamma-Glutamyl Transferase (GGT)": 55.0,
            "Kayser-Fleischer Rings": 1,
            "Neurological Symptoms Score": 3.0,
            "Psychiatric Symptoms": 1,
            "Cognitive Function Score": 78.0,
            "Family History": 1,
            "ATB7B Gene Mutation": 1,
            "Region": "South",
            "Socioeconomic Status": "Medium",
            "Alcohol Use": "False",
            "BMI": 24.5
        }
        if overrides_dict:
            data.update(overrides_dict)
        data.update(kwargs)
        return data

    def test_historical_report_does_not_invoke_ml_or_shap(self):
        """
        Verifies that retrieving a historical assessment report does NOT execute
        the ML prediction ensemble or the SHAP explainer.
        """
        patient = create_or_update_patient({
            "patient_name": "Synthetic Patient Alpha",
            "Age": 32.0,
            "Sex": "Male"
        }, db_path=self.test_db)

        inputs = self._sample_inputs()
        assessment_id = save_assessment(
            patient_id=patient["patient_id"],
            clinical_inputs=inputs,
            prediction_results={
                "prediction_text": "High Predicted Risk – Wilson Disease",
                "prediction_label": 1,
                "probability": 0.8842,
                "model_breakdown": {"svm": 85.0, "logreg": 90.0, "bilstm": 90.2},
                "leipzig_score": 6,
                "shap_waterfall_path": None
            },
            db_path=self.test_db
        )

        with patch("app.safe_shap_and_predict") as mock_predict:
            res = self.client.get(f"/assessment/{assessment_id}/report")
            self.assertEqual(res.status_code, 200)
            mock_predict.assert_not_called()

    def test_historical_report_preserves_specific_assessment_data(self):
        """
        Verifies that historical reports strictly preserve and display the stored
        prediction, probability, clinical inputs, and Leipzig score.
        """
        patient = create_or_update_patient({
            "patient_name": "Synthetic Patient Beta",
            "Age": 45.0,
            "Sex": "Female"
        }, db_path=self.test_db)

        inputs = self._sample_inputs({
            "Age": 45.0,
            "Sex": "Female",
            "ALT": 33.0,
            "Ceruloplasmin Level": 38.0
        })
        assessment_id = save_assessment(
            patient_id=patient["patient_id"],
            clinical_inputs=inputs,
            prediction_results={
                "prediction_text": "Low Predicted Risk – Wilson Disease",
                "prediction_label": 0,
                "probability": 0.1425,
                "model_breakdown": {"svm": 0.12, "logreg": 0.15, "bilstm": 0.157},
                "leipzig_score": 1,
                "shap_waterfall_path": None
            },
            db_path=self.test_db
        )

        res = self.client.get(f"/assessment/{assessment_id}/report")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Stored values verified
        self.assertIn("14.2%", html)  # 0.1425 rounded to 14.2%
        self.assertIn("0.1425", html)
        self.assertIn("12.0%", html)  # SVM breakdown
        self.assertIn("15.0%", html)  # LogReg breakdown
        self.assertIn("Low Predicted Risk – Wilson Disease", html)
        self.assertIn("Synthetic Patient Beta", html)
        self.assertIn(patient["patient_id"], html)
        self.assertIn(f"Archived Assessment #{assessment_id}", html)
        self.assertIn("Original Stored Score: 1 pts", html)

    def test_historical_report_does_not_leak_latest_assessment(self):
        """
        Verifies that requesting an older assessment does not accidentally
        display values from the patient's subsequent or latest assessment.
        """
        patient = create_or_update_patient({
            "patient_name": "Synthetic Patient Longitudinal",
            "Age": 22.0,
            "Sex": "Female"
        }, db_path=self.test_db)

        # Assessment 1: Low risk baseline
        inputs_1 = self._sample_inputs({"ALT": 21.0, "Ceruloplasmin Level": 35.0})
        asmt_id_1 = save_assessment(
            patient_id=patient["patient_id"],
            clinical_inputs=inputs_1,
            prediction_results={
                "prediction_text": "Low Predicted Risk – Wilson Disease",
                "prediction_label": 0,
                "probability": 0.1111,
                "model_breakdown": {"svm": 10.0, "logreg": 12.0, "bilstm": 11.3},
                "leipzig_score": 0,
                "shap_waterfall_path": None
            },
            db_path=self.test_db
        )

        # Assessment 2: High risk exacerbation (Latest)
        inputs_2 = self._sample_inputs({"ALT": 185.0, "Ceruloplasmin Level": 11.0})
        asmt_id_2 = save_assessment(
            patient_id=patient["patient_id"],
            clinical_inputs=inputs_2,
            prediction_results={
                "prediction_text": "High Predicted Risk – Wilson Disease",
                "prediction_label": 1,
                "probability": 0.9456,
                "model_breakdown": {"svm": 94.0, "logreg": 95.0, "bilstm": 94.7},
                "leipzig_score": 7,
                "shap_waterfall_path": None
            },
            db_path=self.test_db
        )

        # Fetch Assessment 1: Must show Assessment 1's values, not Assessment 2's
        res_1 = self.client.get(f"/assessment/{asmt_id_1}/report")
        self.assertEqual(res_1.status_code, 200)
        html_1 = res_1.get_data(as_text=True)

        self.assertIn("11.1%", html_1)
        self.assertIn("0.1111", html_1)
        self.assertIn("Low Predicted Risk", html_1)
        self.assertIn("Original Stored Score: 0 pts", html_1)
        self.assertNotIn("94.6%", html_1)
        self.assertNotIn("0.9456", html_1)
        self.assertNotIn(f"Archived Assessment #{asmt_id_2}", html_1)

        # Fetch Assessment 2: Must show Assessment 2's values
        res_2 = self.client.get(f"/assessment/{asmt_id_2}/report")
        self.assertEqual(res_2.status_code, 200)
        html_2 = res_2.get_data(as_text=True)

        self.assertIn("94.6%", html_2)
        self.assertIn("0.9456", html_2)
        self.assertIn("High Predicted Risk", html_2)
        self.assertIn("Original Stored Score: 7 pts", html_2)
        self.assertNotIn("11.1%", html_2)
        self.assertNotIn("0.1111", html_2)

    def test_missing_historical_fields_handled_safely_and_labeled(self):
        """
        Verifies that assessments with missing clinical inputs, null model breakdowns,
        or null Leipzig score render safely without crashing.
        """
        patient = create_or_update_patient({
            "patient_name": "Synthetic Patient Sparse",
            "Age": 30.0,
            "Sex": "Male"
        }, db_path=self.test_db)

        # Inputs with multiple None/null values
        sparse_inputs = {
            "Age": 30.0,
            "Sex": "Male",
            "Ceruloplasmin Level": None,
            "Copper in Blood Serum": None,
            "ALT": 45.0,
            "AST": None,
            "Region": None
        }
        asmt_id = save_assessment(
            patient_id=patient["patient_id"],
            clinical_inputs=sparse_inputs,
            prediction_results={
                "prediction_text": "Low Predicted Risk – Wilson Disease",
                "prediction_label": 0,
                "probability": 0.25,
                "model_breakdown": {"svm": None, "logreg": 25.0, "bilstm": None},
                "leipzig_score": None,
                "shap_waterfall_path": None
            },
            db_path=self.test_db
        )

        res = self.client.get(f"/assessment/{asmt_id}/report")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn("Synthetic Patient Sparse", html)
        self.assertIn("25.0%", html)
        # Should render without throwing 500 or template syntax error
        self.assertIn("Clinical Interpretation Notice", html)

    def test_missing_or_tampered_shap_artifact_safe_fallback(self):
        """
        Verifies that non-existent or path-traversal SHAP artifact paths
        render a graceful fallback without breaking the report or exposing files.
        """
        patient = create_or_update_patient({
            "patient_name": "Synthetic Patient Missing SHAP",
            "Age": 29.0,
            "Sex": "Female"
        }, db_path=self.test_db)

        # Save with non-existent path
        asmt_id_1 = save_assessment(
            patient_id=patient["patient_id"],
            clinical_inputs=self._sample_inputs(),
            prediction_results={
                "prediction_text": "Low Predicted Risk – Wilson Disease",
                "prediction_label": 0,
                "probability": 0.15,
                "shap_waterfall_path": "plots/non_existent_shap_test_12345.png"
            },
            db_path=self.test_db
        )

        res_1 = self.client.get(f"/assessment/{asmt_id_1}/report")
        self.assertEqual(res_1.status_code, 200)
        html_1 = res_1.get_data(as_text=True)
        self.assertIn("SHAP feature attribution waterfall plot was not recorded or is no longer available on disk", html_1)
        self.assertNotIn("non_existent_shap_test_12345.png", html_1)

        # Save with path traversal attempt
        asmt_id_2 = save_assessment(
            patient_id=patient["patient_id"],
            clinical_inputs=self._sample_inputs(),
            prediction_results={
                "prediction_text": "Low Predicted Risk – Wilson Disease",
                "prediction_label": 0,
                "probability": 0.15,
                "shap_waterfall_path": "../../secret_config.ini"
            },
            db_path=self.test_db
        )

        res_2 = self.client.get(f"/assessment/{asmt_id_2}/report")
        self.assertEqual(res_2.status_code, 200)
        html_2 = res_2.get_data(as_text=True)
        self.assertIn("SHAP feature attribution waterfall plot was not recorded or is no longer available on disk", html_2)
        self.assertNotIn("secret_config.ini", html_2)

    def test_print_styling_and_ui_controls_hidden(self):
        """
        Verifies print CSS rules exist and hide navigation, action buttons,
        collapsible controls, and chat widgets. Also verifies button text and disclaimer.
        """
        patient = create_or_update_patient({
            "patient_name": "Synthetic Patient Print Test",
            "Age": 50.0,
            "Sex": "Male"
        }, db_path=self.test_db)

        asmt_id = save_assessment(
            patient_id=patient["patient_id"],
            clinical_inputs=self._sample_inputs(),
            prediction_results={
                "prediction_text": "Low Predicted Risk – Wilson Disease",
                "prediction_label": 0,
                "probability": 0.20,
                "leipzig_score": 2
            },
            db_path=self.test_db
        )

        res = self.client.get(f"/assessment/{asmt_id}/report")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # 1. Print CSS Media Query
        self.assertIn("@media print", html)
        self.assertIn(".navbar-custom", html)
        self.assertIn(".web-action-controls", html)
        self.assertIn(".web-collapsible-controls", html)
        self.assertIn("#chat-toggle", html)
        self.assertIn("#chat-widget", html)

        # 2. Button Labeling: Accurately described as Print Report / Save as PDF
        self.assertIn("Print Report / Save as PDF", html)

        # 3. Patient Identity card in report
        self.assertIn('id="card-patient-identity"', html)
        self.assertIn("Synthetic Patient Print Test", html)
        self.assertIn(patient["patient_id"], html)

        # 4. Mandatory Clinical Decision Support Disclaimer
        self.assertIn("clinical-disclaimer-box", html)
        self.assertIn("Mandatory Clinical Decision Support Notice & Disclaimer", html)
        self.assertIn("not a confirmed medical diagnosis", html)

        # 5. Integrity notices regarding recalculated values
        self.assertIn("recalculated using the current application logic", html)


if __name__ == "__main__":
    unittest.main()
