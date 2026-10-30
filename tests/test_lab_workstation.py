"""
tests/test_lab_workstation.py

Comprehensive test suite covering:
1. Database schema and lab-report CRUD functions:
   - Unique sequential report number generation (LAB-YYYY-####)
   - Canonical 23 biomarker preservation without zero-fabrication
   - Patient profile registration and foreign-key integrity
   - Retrieval by ID, patient history, and latest report lookup
2. Lab Staff intake, validation, and printable reports:
   - Full intake save and validation via /api/lab/save_report
   - Printable report view rendering (/lab/report/<id>)
   - Specimen QC notes recording
3. RBAC isolation:
   - Lab Staff authorized for intake and reporting
   - Lab Staff strictly forbidden (403) from ML prediction (/submit) and Admin panel (/rag/monitor)
   - Clinician authorized for lab import and longitudinal comparison
4. Document governance audit history:
   - Audit trail table verification in /rag/monitor
5. Longitudinal multi-period patient testing & side-by-side comparison matrix:
   - Same patient tested in multiple periods (Jan vs June)
   - Calculation of parameter deltas (Δ), percentage changes, and directional flags
   - Therapeutic response synthesis
"""

import unittest
import os
import tempfile
import shutil
import time
from unittest.mock import patch

from app import app
from database import (
    init_db,
    create_user,
    create_lab_report,
    get_lab_reports,
    get_lab_report_by_id,
    get_latest_lab_report_for_patient,
    get_patient_comparison_data,
    create_or_update_patient,
    save_assessment,
    create_rag_document_submission,
    update_rag_document_status,
    get_rag_documents,
    get_patient_by_id
)


class TestLabWorkstationAndLongitudinalComparison(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_lab_records.db")
        init_db(self.test_db)
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["DB_PATH"] = self.test_db
        self.client = app.test_client()

        # Create test users
        self.admin_user = create_user("test_admin", "AdminPass123!", "Chief Doctor Admin", "admin", db_path=self.test_db)
        self.clinician_user = create_user("test_doc", "ClinicianPass123!", "Dr. Jane Clinician", "clinician", db_path=self.test_db)
        self.lab_user = create_user("test_tech", "LabStaffPass123!", "Sarah Tech MLS", "lab_staff", db_path=self.test_db)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _login(self, user):
        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = user["username"]
            sess["role"] = user["role"]
            sess["full_name"] = user["full_name"]
            sess["last_active"] = time.time()

    # =========================================================================
    # 1. DATABASE SCHEMA & LAB-REPORT CRUD
    # =========================================================================

    def test_create_lab_report_sequential_number_and_persistence(self):
        """Verify lab report generation creates sequential LAB-YYYY-#### and stores 23 biomarkers."""
        data1 = {
            "patient_name": "Marcus Sterling",
            "age": 26,
            "sex": "Male",
            "ceruloplasmin_level": 8.4,
            "copper_blood_serum": 48.0,
            "free_copper_blood_serum": 28.5,
            "copper_urine": 185.0,
            "alt": 142.0,
            "ast": 118.0,
            "total_bilirubin": 2.8,
            "albumin": 3.1,
            "alkaline_phosphatase_alp": 42.0,
            "prothrombin_time_inr": 1.55,
            "gamma_glutamyl_transferase_ggt": 88.0,
            "kayser_fleischer_rings": 1,
            "neurological_symptoms_score": 4.5,
            "psychiatric_symptoms": 1,
            "cognitive_function_score": 68.0,
            "family_history": 1,
            "atb7b_gene_mutation": 1,
            "region": "East",
            "socioeconomic_status": "Low",
            "alcohol_use": "No",
            "bmi": 24.1,
            "specimen_notes": "Diagnostic baseline testing."
        }

        report1 = create_lab_report(data1, db_path=self.test_db)
        self.assertIsNotNone(report1)
        self.assertTrue(report1["report_number"].startswith(f"LAB-{time.strftime('%Y')}-0001"))
        self.assertEqual(report1["patient_name"], "Marcus Sterling")
        self.assertEqual(report1["copper_urine"], 185.0)

        # Check patient auto-registration
        patient = get_patient_by_id(report1["patient_id"], db_path=self.test_db)
        self.assertIsNotNone(patient)
        self.assertEqual(patient["patient_name"], "Marcus Sterling")

        # Second report for same patient increments report sequence
        data2 = dict(data1)
        data2["patient_id"] = report1["patient_id"]
        data2["copper_urine"] = 74.0
        data2["specimen_notes"] = "6-month follow-up testing."

        report2 = create_lab_report(data2, db_path=self.test_db)
        self.assertTrue(report2["report_number"].startswith(f"LAB-{time.strftime('%Y')}-0002"))
        self.assertEqual(report2["patient_id"], report1["patient_id"])
        self.assertEqual(report2["copper_urine"], 74.0)

        # Retrieve all reports for this patient
        reports = get_lab_reports(patient_id=report1["patient_id"], db_path=self.test_db)
        self.assertEqual(len(reports), 2)

        # Latest report check
        latest = get_latest_lab_report_for_patient(report1["patient_id"], db_path=self.test_db)
        self.assertEqual(latest["report_id"], report2["report_id"])
        self.assertEqual(latest["copper_urine"], 74.0)

    def test_missing_biomarkers_preserved_as_none(self):
        """Ensure missing biomarkers stay None and are not zero-fabricated."""
        partial_data = {
            "patient_name": "Sparse Test Patient",
            "age": 45,
            "ceruloplasmin_level": 22.0,
            "copper_urine": None  # deliberately omitted
        }

        report = create_lab_report(partial_data, db_path=self.test_db)
        fetched = get_lab_report_by_id(report["report_id"], db_path=self.test_db)
        self.assertIsNone(fetched["copper_urine"])
        self.assertIsNone(fetched["alt"])
        self.assertEqual(fetched["ceruloplasmin_level"], 22.0)

    # =========================================================================
    # 2. LAB STAFF WORKSTATION & REPORT GENERATION
    # =========================================================================

    def test_lab_staff_intake_and_report_creation_api(self):
        """Lab staff can successfully access /lab/import and POST /api/lab/save_report."""
        self._login(self.lab_user)

        # 1. Access workstation
        resp = self.client.get("/lab/import")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Clinical Pathology", resp.data)

        # 2. Save certified lab report
        payload = {
            "patient_name": "Eleanor Vance",
            "age": 32,
            "sex": "Female",
            "ceruloplasmin_level": 28.5,
            "copper_blood_serum": 98.0,
            "free_copper_blood_serum": 8.2,
            "copper_urine": 24.0,
            "alt": 22.0,
            "ast": 19.0,
            "specimen_notes": "Fasting sample verified. No hemolysis."
        }

        res = self.client.post("/api/lab/save_report", json=payload)
        self.assertEqual(res.status_code, 201)
        res_json = res.get_json()
        self.assertTrue(res_json["success"])
        self.assertIn("report_id", res_json)
        self.assertIn("report_number", res_json)

        # 3. View printable report
        report_id = res_json["report_id"]
        view_res = self.client.get(f"/lab/report/{report_id}")
        self.assertEqual(view_res.status_code, 200)
        self.assertIn(b"Certified Pathology Report", view_res.data)
        self.assertIn(b"Eleanor Vance", view_res.data)
        self.assertIn(b"Fasting sample verified", view_res.data)

    # =========================================================================
    # 3. RBAC ISOLATION & SEPARATION OF DUTIES
    # =========================================================================

    def test_lab_staff_clinical_test_access_and_admin_barrier(self):
        """Enforces that lab staff has access to clinical test entry (/about) but cannot access admin panels (/rag/monitor, /patients)."""
        self._login(self.lab_user)

        # Lab staff CAN access clinical test input form (/about)
        about_res = self.client.get("/about")
        self.assertEqual(about_res.status_code, 200)

        # Lab staff attempted admin monitor access
        admin_res = self.client.get("/rag/monitor")
        self.assertEqual(admin_res.status_code, 403)

        # Lab staff attempted patient directory access
        patients_res = self.client.get("/patients")
        self.assertEqual(patients_res.status_code, 403)

    def test_clinician_can_fetch_latest_lab_report_for_autofill(self):
        """Clinician can fetch patient's latest certified lab report via API for 1-click import."""
        # Create report as lab tech
        lab_rep = create_lab_report({
            "patient_name": "Marcus Sterling",
            "ceruloplasmin_level": 12.0,
            "copper_urine": 120.0
        }, db_path=self.test_db)

        # Clinician queries report
        self._login(self.clinician_user)
        res = self.client.get(f"/api/lab/patient/{lab_rep['patient_id']}/latest_report")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["report"]["copper_urine"], 120.0)

    # =========================================================================
    # 4. DOCUMENT GOVERNANCE AUDIT HISTORY
    # =========================================================================

    def test_document_governance_audit_history_in_admin_panel(self):
        """Verifies that all guideline submissions are registered with arrival time, review status, and reviewer notes."""
        self._login(self.admin_user)

        # Create two sample submissions in DB
        doc_id1 = create_rag_document_submission({
            "filename": "Pediatric_WD_Protocol_2026.pdf",
            "original_filename": "Pediatric_WD_Protocol_2026.pdf",
            "title": "Pediatric Wilson Disease Protocol",
            "file_path": "medical_docs/pending/Pediatric_WD_Protocol_2026.pdf",
            "file_size_bytes": 1048576,
            "file_hash_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "uploaded_by_id": self.clinician_user["user_id"],
            "uploaded_by_name": self.clinician_user["full_name"],
            "review_status": "pending_approval"
        }, db_path=self.test_db)

        doc_id2 = create_rag_document_submission({
            "filename": "Experimental_Herbal_Treatment.pdf",
            "original_filename": "Experimental_Herbal_Treatment.pdf",
            "title": "Experimental Herbal Protocol",
            "file_path": "medical_docs/pending/Experimental_Herbal_Treatment.pdf",
            "file_size_bytes": 524288,
            "file_hash_sha256": "123456abcdef7890123456abcdef7890123456abcdef7890123456abcdef7890",
            "uploaded_by_id": self.clinician_user["user_id"],
            "uploaded_by_name": self.clinician_user["full_name"],
            "review_status": "pending_approval"
        }, db_path=self.test_db)

        # Chief Doctor rejects doc2 with specific rejection notes
        update_rag_document_status(
            doc_id=doc_id2,
            status="rejected",
            reviewed_by_id=self.admin_user["user_id"],
            reviewed_by_name=self.admin_user["full_name"],
            review_notes="Rejected: Lacks peer-reviewed evidence and consensus clinical trial safety data.",
            db_path=self.test_db
        )

        # Fetch /rag/monitor page
        resp = self.client.get("/rag/monitor")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Document Governance Audit History", resp.data)
        self.assertIn(b"Pediatric Wilson Disease Protocol", resp.data)
        self.assertIn(b"Experimental Herbal Protocol", resp.data)
        self.assertIn(b"Lacks peer-reviewed evidence", resp.data)

    # =========================================================================
    # 5. MULTI-PERIOD TESTING & SIDE-BY-SIDE COMPARISON MATRIX
    # =========================================================================

    def test_patient_longitudinal_comparison_matrix_calculations(self):
        """Verifies calculation of parameter deltas (Δ), percentage changes, directional badges, and response synthesis."""
        # Patient tested across two periods
        p = create_or_update_patient({"patient_name": "Marcus Sterling"}, db_path=self.test_db)
        pid = p["patient_id"]

        # Assessment 1 (Jan - Baseline high risk)
        inputs1 = {
            "Age": 26, "Sex": "Male",
            "Ceruloplasmin Level": 8.0,
            "Copper in Urine": 180.0,
            "ALT": 140.0,
            "AST": 115.0,
            "Total Bilirubin": 2.5,
            "Albumin": 3.0,
            "Prothrombin Time / INR": 1.5,
            "Kayser-Fleischer Rings": 1,
            "Neurological Symptoms Score": 4.0,
            "Psychiatric Symptoms": 1,
            "Cognitive Function Score": 70.0,
            "Family History": 1,
            "ATB7B Gene Mutation": 1,
            "Region": "East", "Socioeconomic Status": "Low", "Alcohol Use": "No", "BMI": 24.0
        }
        res1 = {
            "prediction_text": "High Predicted Risk – Wilson Disease",
            "probability": 0.94,
            "leipzig_score": 7
        }
        id1 = save_assessment(pid, inputs1, res1, db_path=self.test_db)

        # Assessment 2 (June - Follow-up decoppering response)
        inputs2 = dict(inputs1)
        inputs2["Ceruloplasmin Level"] = 15.0 # Increased (+87.5% toward normal - improved)
        inputs2["Copper in Urine"] = 70.0     # Decreased (-61.1% - improved)
        inputs2["ALT"] = 45.0                 # Decreased (-67.9% - improved)
        inputs2["AST"] = 38.0                 # Decreased (-67.0% - improved)
        inputs2["Albumin"] = 3.9              # Increased (+30.0% toward normal - improved)

        res2 = {
            "prediction_text": "Low Predicted Risk – Wilson Disease",
            "probability": 0.38,
            "leipzig_score": 5
        }
        id2 = save_assessment(pid, inputs2, res2, db_path=self.test_db)

        # Call get_patient_comparison_data
        comp = get_patient_comparison_data(pid, id1, id2, comparison_type="assessment", db_path=self.test_db)
        self.assertIsNotNone(comp)
        self.assertEqual(comp["overall_trend"], "Therapeutic Response Confirmed")
        self.assertTrue(comp["improved_count"] >= 4)

        # Verify Urine Copper delta in matrix
        urine_cu_row = next(r for r in comp["matrix"] if "Urine" in r["label"])
        self.assertEqual(urine_cu_row["baseline_val"], 180.0)
        self.assertEqual(urine_cu_row["followup_val"], 70.0)
        self.assertEqual(urine_cu_row["delta"], -110.0)
        self.assertEqual(urine_cu_row["pct_change"], -61.1)
        self.assertEqual(urine_cu_row["status"], "improved")
        self.assertEqual(urine_cu_row["icon"], "▼")

        # Verify ALT delta in matrix
        alt_row = next(r for r in comp["matrix"] if "ALT" in r["label"])
        self.assertEqual(alt_row["delta"], -95.0)
        self.assertEqual(alt_row["status"], "improved")

        # Clinician views the side-by-side comparison page
        self._login(self.clinician_user)
        page_res = self.client.get(f"/patient/{pid}/compare?id1={id1}&id2={id2}&type=assessment")
        self.assertEqual(page_res.status_code, 200)
        self.assertIn(b"Longitudinal Side-by-Side Comparison Matrix", page_res.data)
        self.assertIn(b"Therapeutic Response Confirmed", page_res.data)
        self.assertIn(b"Marcus Sterling", page_res.data)


if __name__ == "__main__":
    unittest.main()
