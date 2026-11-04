"""
tests/test_lab_import.py

Automated test suite for Stage 4:
- Laboratory data upload (CSV & JSON)
- Schema normalization & 23-feature alignment
- Security checks: file extensions, size limits (>2 MB), malformed files, XSS safety
- Duplicate & conflicting column detection
- Physiological bounds & categorical validation
- Preservation of missing clinical values (no zero-filling or fabrication)
- Patient identity isolation from the 23-feature ML input vector
- Template download endpoints
- Import preview non-prediction isolation
"""

import unittest
import json
import io
import os
import tempfile
import shutil

from app import app
from database import init_db, get_patient_by_id, get_assessment_by_id, open_db
from lab_import import (
    parse_lab_file_content,
    generate_sample_csv_text,
    generate_sample_json_text,
    generate_sample_docx_bytes,
    CANONICAL_FEATURES,
    MAX_UPLOAD_SIZE
)


class TestLaboratoryDataImport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_lab_records.db")
        init_db(self.test_db)
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["DB_PATH"] = self.test_db
        self.client = app.test_client()

        from database import create_user
        import time
        user = create_user("test_labstaff", "LabStaffPass123!", "Lab Specialist", "lab_staff", db_path=self.test_db)
        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = user["username"]
            sess["role"] = user["role"]
            sess["full_name"] = user["full_name"]
            sess["last_active"] = time.time()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_csv_upload(self):
        """Tests uploading a valid complete CSV file."""
        csv_data = generate_sample_csv_text().encode("utf-8")
        data = {
            "file": (io.BytesIO(csv_data), "patient_lab_results.csv")
        }
        res = self.client.post("/api/upload_lab_file", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["format"], "csv")
        self.assertEqual(json_data["patient_name"], "Sample Patient (Fictional)")
        self.assertEqual(json_data["patient_id"], "NEUROMED001")
        self.assertEqual(json_data["imported_count"], len(CANONICAL_FEATURES))
        self.assertEqual(len(json_data["missing_fields"]), 0)
        self.assertEqual(len(json_data["rejected_fields"]), 0)
        self.assertIn("inputCeruloplasmin", json_data["form_field_map"])
        self.assertEqual(json_data["form_field_map"]["inputCeruloplasmin"], 28.5)

    def test_valid_json_upload(self):
        """Tests uploading a valid complete JSON file."""
        json_data = generate_sample_json_text().encode("utf-8")
        data = {
            "file": (io.BytesIO(json_data), "patient_lab_results.json")
        }
        res = self.client.post("/api/upload_lab_file", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        resp_json = res.get_json()
        self.assertTrue(resp_json["success"])
        self.assertEqual(resp_json["format"], "json")
        self.assertEqual(resp_json["patient_name"], "Sample Patient (Fictional)")
        self.assertEqual(resp_json["patient_id"], "NEUROMED001")
        self.assertEqual(resp_json["imported_count"], len(CANONICAL_FEATURES))
        self.assertEqual(len(resp_json["missing_fields"]), 0)
        self.assertEqual(resp_json["form_field_map"]["inputAST"], 25.0)

    def test_unsupported_file_extension(self):
        """Verifies rejection of unsupported file formats."""
        data = {
            "file": (io.BytesIO(b"binary executable data"), "report.exe")
        }
        res = self.client.post("/api/upload_lab_file", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data["success"])
        self.assertIn("Unsupported file format", json_data["error"])

    def test_valid_docx_upload(self):
        """Tests uploading a valid complete Word (.docx) laboratory report."""
        docx_bytes = generate_sample_docx_bytes()
        data = {
            "file": (io.BytesIO(docx_bytes), "patient_lab_results.docx")
        }
        res = self.client.post("/api/upload_lab_file", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        resp_json = res.get_json()
        self.assertTrue(resp_json["success"])
        self.assertEqual(resp_json["format"], "docx")
        self.assertEqual(resp_json["patient_name"], "Sample Patient (Fictional)")
        self.assertEqual(resp_json["patient_id"], "NEUROMED001")
        self.assertEqual(resp_json["imported_count"], len(CANONICAL_FEATURES))
        self.assertEqual(len(resp_json["missing_fields"]), 0)
        self.assertEqual(resp_json["form_field_map"]["inputCeruloplasmin"], 28.5)
        self.assertEqual(resp_json["form_field_map"]["inputALT"], 22.0)

    def test_valid_pdf_upload(self):
        """Tests uploading a valid PDF clinical laboratory report."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            report_text = (
                "Patient Full Name: Eleanor Vance\n"
                "Patient ID: WDP-4401\n"
                "Age: 32 years\n"
                "Sex: Female\n"
                "Geographic Region: North\n"
                "Socioeconomic Status: Medium\n"
                "BMI: 21.8 kg/m2\n"
                "Alcohol Use: False\n"
                "Serum Ceruloplasmin Level: 12.0 mg/dL\n"
                "Total Serum Copper: 130.0 ug/dL\n"
                "Free Copper in Blood Serum: 16.0 ug/dL\n"
                "24h Urinary Copper Excretion: 55.0 ug/24h\n"
                "Alanine Aminotransferase (ALT): 45.0 U/L\n"
                "Aspartate Aminotransferase (AST): 50.0 U/L\n"
                "Total Bilirubin: 1.2 mg/dL\n"
                "Serum Albumin: 3.9 g/dL\n"
                "Alkaline Phosphatase (ALP): 85.0 U/L\n"
                "Prothrombin Time / INR: 1.10\n"
                "Gamma-Glutamyl Transferase (GGT): 30.0 U/L\n"
                "Kayser-Fleischer Rings: Present\n"
                "Neurological Symptoms Score: 3.0 / 10\n"
                "Psychiatric Symptoms: Present\n"
                "Cognitive Function Score: 80.0 / 100\n"
                "Family History: Positive\n"
                "ATB7B Gene Mutation: Mutation Present"
            )
            ax.text(0.05, 0.95, report_text, verticalalignment='top', fontsize=10, family='monospace')
            pdf.savefig(fig)
            plt.close(fig)

        data = {
            "file": (io.BytesIO(pdf_buf.getvalue()), "clinical_report.pdf")
        }
        res = self.client.post("/api/upload_lab_file", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        resp_json = res.get_json()
        self.assertTrue(resp_json["success"])
        self.assertEqual(resp_json["format"], "pdf")
        self.assertEqual(resp_json["patient_name"], "Eleanor Vance")
        self.assertEqual(resp_json["patient_id"], "WDP-4401")
        self.assertEqual(resp_json["imported_count"], len(CANONICAL_FEATURES))
        self.assertEqual(resp_json["form_field_map"]["inputCeruloplasmin"], 12.0)
        self.assertEqual(resp_json["form_field_map"]["inputALT"], 45.0)
        self.assertEqual(resp_json["form_field_map"]["inputKFR"], "Yes")

    def test_partial_document_strictly_preserves_missing_fields(self):
        """Verifies partial PDF/DOCX strictly preserves unmeasured fields as missing (no zero fabrication)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            partial_text = (
                "Patient Name: Partial Record Patient\n"
                "Serum Ceruloplasmin: 15.5 mg/dL\n"
                "Total Serum Copper: 110.0 ug/dL\n"
                "ALT: 28.0 U/L\n"
                "AST: 32.0 U/L"
            )
            ax.text(0.05, 0.95, partial_text, verticalalignment='top')
            pdf.savefig(fig)
            plt.close(fig)

        data = {
            "file": (io.BytesIO(pdf_buf.getvalue()), "partial_panel.pdf")
        }
        res = self.client.post("/api/upload_lab_file", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        resp_json = res.get_json()
        self.assertTrue(resp_json["success"])
        self.assertEqual(resp_json["imported_count"], 4)
        self.assertEqual(len(resp_json["missing_fields"]), len(CANONICAL_FEATURES) - 4)
        self.assertIn("Total Bilirubin", resp_json["missing_fields"])
        self.assertIn("Kayser-Fleischer Rings", resp_json["missing_fields"])
        self.assertNotIn("inputTotalBilirubin", resp_json["form_field_map"])
        self.assertNotIn("inputKFR", resp_json["form_field_map"])

    def test_malformed_and_empty_pdf_and_docx(self):
        """Verifies rejection of corrupt and empty PDF or Word files."""
        # Corrupt PDF
        res1 = self.client.post(
            "/api/upload_lab_file",
            data={"file": (io.BytesIO(b"%PDF-1.4 corrupt content not a valid xref"), "broken.pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res1.status_code, 400)
        self.assertFalse(res1.get_json()["success"])

        # Corrupt DOCX
        res2 = self.client.post(
            "/api/upload_lab_file",
            data={"file": (io.BytesIO(b"PK\x03\x04 corrupt zip archive"), "broken.docx")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res2.status_code, 400)
        self.assertFalse(res2.get_json()["success"])

    def test_malformed_csv_and_json(self):
        """Verifies rejection of malformed or unparseable files."""
        # Empty CSV
        data_empty = {
            "file": (io.BytesIO(b""), "empty.csv")
        }
        res1 = self.client.post("/api/upload_lab_file", data=data_empty, content_type="multipart/form-data")
        self.assertEqual(res1.status_code, 400)
        self.assertFalse(res1.get_json()["success"])

        # Corrupt JSON
        data_corrupt_json = {
            "file": (io.BytesIO(b"{broken json content"), "bad.json")
        }
        res2 = self.client.post("/api/upload_lab_file", data=data_corrupt_json, content_type="multipart/form-data")
        self.assertEqual(res2.status_code, 400)
        self.assertFalse(res2.get_json()["success"])
        self.assertIn("Malformed JSON", res2.get_json()["error"])

    def test_oversized_upload_rejection(self):
        """Verifies enforcement of the 2 MB maximum upload size limit."""
        oversized_bytes = b"Age,Sex\n24,Female\n" + (b"X" * (MAX_UPLOAD_SIZE + 500))
        data = {
            "file": (io.BytesIO(oversized_bytes), "oversized.csv")
        }
        res = self.client.post("/api/upload_lab_file", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 413)
        self.assertFalse(res.get_json()["success"])
        self.assertIn("exceeds", res.get_json()["error"])

    def test_duplicate_column_headers_rejection(self):
        """Tests that duplicate/conflicting column headers are detected and rejected."""
        # Literal duplicate headers in CSV
        dup_csv = "Age,Sex,ALT,ALT\n24,Female,35,40\n".encode("utf-8")
        res = parse_lab_file_content(dup_csv, "duplicate.csv")
        self.assertFalse(res["success"])
        self.assertIn("duplicate column headers", res["error"].lower())

    def test_conflicting_alias_columns(self):
        """Tests that conflicting alias mappings for the same feature are caught."""
        # JSON with both 'serum_copper' and 'Copper in Blood Serum' having different values
        conflicting_json = json.dumps({
            "serum_copper": 120.0,
            "Copper in Blood Serum": 250.0
        }).encode("utf-8")
        res = parse_lab_file_content(conflicting_json, "conflict.json")
        self.assertTrue(res["success"])
        self.assertIn("Copper in Blood Serum", res["rejected_fields"])
        self.assertIn("Conflicting duplicate", res["rejected_fields"]["Copper in Blood Serum"])

    def test_invalid_bounds_and_categoricals(self):
        """Verifies that out-of-range numerical values and invalid categoricals are placed in rejected_fields."""
        lab_dict = {
            "Age": 250,              # Beyond acceptable range 1-120
            "Sex": "Alien",          # Invalid categorical
            "Region": "Atlantis",    # Invalid region
            "Cognitive Function Score": 15.0,  # Below scale 30-100
            "Ceruloplasmin Level": 25.0        # Valid
        }
        res = parse_lab_file_content(json.dumps(lab_dict).encode("utf-8"), "test.json")
        self.assertTrue(res["success"])
        self.assertIn("Age", res["rejected_fields"])
        self.assertIn("Sex", res["rejected_fields"])
        self.assertIn("Region", res["rejected_fields"])
        self.assertIn("Cognitive Function Score", res["rejected_fields"])
        self.assertIn("Ceruloplasmin Level", res["imported_fields"])
        self.assertEqual(res["imported_fields"]["Ceruloplasmin Level"], 25.0)

    def test_missing_values_preserved_as_missing(self):
        """Verifies that missing clinical parameters are strictly preserved and NEVER zero-filled."""
        partial_data = {
            "Age": 30,
            "Sex": "Male",
            "Ceruloplasmin Level": None,
            "ALT": "",
            "AST": "N/A"
        }
        res = parse_lab_file_content(json.dumps(partial_data).encode("utf-8"), "partial.json")
        self.assertTrue(res["success"])
        # Only Age and Sex should be imported
        self.assertIn("Age", res["imported_fields"])
        self.assertIn("Sex", res["imported_fields"])
        self.assertNotIn("Ceruloplasmin Level", res["imported_fields"])
        self.assertNotIn("ALT", res["imported_fields"])
        self.assertNotIn("AST", res["imported_fields"])
        # The unprovided features must be in missing_fields
        self.assertIn("Ceruloplasmin Level", res["missing_fields"])
        self.assertIn("ALT", res["missing_fields"])
        self.assertIn("AST", res["missing_fields"])
        # And NOT zero in the form_field_map
        self.assertNotIn("inputCeruloplasmin", res["form_field_map"])
        self.assertNotIn("inputALT", res["form_field_map"])

    def test_patient_identity_isolation(self):
        """Verifies patient identity (Name, ID) is kept outside the 23 clinical features."""
        data_with_patient = {
            "Patient Name": "Alexander Hamilton",
            "Patient ID": "NEUROMED777",
            "Age": 35,
            "Sex": "Male"
        }
        res = parse_lab_file_content(json.dumps(data_with_patient).encode("utf-8"), "patient.json")
        self.assertEqual(res["patient_name"], "Alexander Hamilton")
        self.assertEqual(res["patient_id"], "NEUROMED777")
        # Ensure neither patient_name nor patient_id is inside imported_fields
        self.assertNotIn("_patient_name", res["imported_fields"])
        self.assertNotIn("_patient_id", res["imported_fields"])
        self.assertNotIn("Patient Name", res["imported_fields"])
        self.assertNotIn("Patient ID", res["imported_fields"])
        # All items in imported_fields must belong to CANONICAL_FEATURES
        for k in res["imported_fields"].keys():
            self.assertIn(k, CANONICAL_FEATURES)

    def test_upload_does_not_trigger_prediction_or_database_write(self):
        """Verifies that /api/upload_lab_file is purely an inspection endpoint and does not create DB records."""
        json_data = generate_sample_json_text().encode("utf-8")
        data = {
            "file": (io.BytesIO(json_data), "inspect_only.json")
        }
        res = self.client.post("/api/upload_lab_file", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)

        # Check database: should have 0 patients and 0 assessments
        with open_db(self.test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM patients")
            self.assertEqual(cursor.fetchone()[0], 0)
            cursor.execute("SELECT COUNT(*) FROM assessments")
            self.assertEqual(cursor.fetchone()[0], 0)

    def test_sample_template_endpoints(self):
        """Tests that sample CSV and JSON templates are downloadable."""
        # CSV download
        res_csv = self.client.get("/api/sample_template?format=csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertEqual(res_csv.mimetype, "text/csv")
        self.assertIn("attachment; filename=wilson_lab_template.csv", res_csv.headers.get("Content-Disposition", ""))
        self.assertIn("Ceruloplasmin Level", res_csv.get_data(as_text=True))

        # JSON download
        res_json = self.client.get("/api/sample_template?format=json")
        self.assertEqual(res_json.status_code, 200)
        self.assertEqual(res_json.mimetype, "application/json")
        self.assertIn("attachment; filename=wilson_lab_template.json", res_json.headers.get("Content-Disposition", ""))
        parsed = json.loads(res_json.get_data(as_text=True))
        self.assertEqual(parsed["Patient ID"], "NEUROMED001")

        # DOCX download
        res_docx = self.client.get("/api/sample_template?format=docx")
        self.assertEqual(res_docx.status_code, 200)
        self.assertEqual(res_docx.mimetype, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        self.assertIn("attachment; filename=wilson_lab_template.docx", res_docx.headers.get("Content-Disposition", ""))
        self.assertGreater(len(res_docx.data), 1000)


if __name__ == "__main__":
    unittest.main()
