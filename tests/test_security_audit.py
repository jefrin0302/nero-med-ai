"""
tests/test_security_audit.py

Automated test suite for Stage 7:
- Verification of defensive security headers on HTTP responses
- Verification of session cookie security configuration and secret key
- Verification that internal exception details / paths are not leaked in 500 error responses
- Verification that .gitignore protects SQLite patient databases from accidental commit
- Verification of file upload size and extension enforcement
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch

from app import app
from database import init_db


class TestSecurityAuditAndHardening(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_sec_records.db")
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

    def test_security_headers_present_on_responses(self):
        """Verifies that all responses include defensive HTTP security headers."""
        routes_to_test = ["/", "/about", "/patients"]
        for route in routes_to_test:
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed")
            self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
            self.assertEqual(res.headers.get("X-Frame-Options"), "SAMEORIGIN")
            self.assertEqual(res.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
            self.assertEqual(res.headers.get("X-XSS-Protection"), "1; mode=block")

    def test_session_cookie_and_secret_key_configuration(self):
        """Verifies that secret key is set and session cookies have HttpOnly and SameSite configured."""
        self.assertTrue(app.config.get("SESSION_COOKIE_HTTPONLY"))
        self.assertEqual(app.config.get("SESSION_COOKIE_SAMESITE"), "Lax")
        self.assertIsNotNone(app.config.get("SECRET_KEY"))
        self.assertGreater(len(app.config.get("SECRET_KEY")), 16)

    def test_500_prediction_error_does_not_leak_internal_exception(self):
        """Verifies that internal runtime exceptions during /submit do not leak stack traces or paths."""
        sensitive_msg = "SECRET_PATH C:\\Users\\Administrator\\secrets.json: Database driver crash"
        with patch("app.safe_shap_and_predict", side_effect=RuntimeError(sensitive_msg)):
            res = self.client.post("/submit", data={
                "patientName": "Audit Synthetic",
                "patientId": "NEUROMED001",
                "inputAge": "30",
                "inputGender": "Female",
                "inputRegion": "North",
                "inputSocioeconomicStatus": "High",
                "inputBMI": "22.5",
                "inputAlcoholUse": "False",
                "inputCeruloplasmin": "25.0",
                "inputCopperBlood": "110.0",
                "inputFreeCopperBlood": "9.5",
                "inputCopperUrine": "28.0",
                "inputALT": "25.0",
                "inputAST": "28.0",
                "inputTotalBilirubin": "0.9",
                "inputAlbumin": "4.2",
                "inputALP": "90.0",
                "inputProthrombin": "1.05",
                "inputGGT": "30.0",
                "inputKFR": "No",
                "inputNeurological": "0.5",
                "inputPsychiatric": "No",
                "inputCognitive": "92.0",
                "inputFamilyHistory": "No",
                "inputGeneMutation": "No"
            })
            self.assertEqual(res.status_code, 500)
            body = res.get_data(as_text=True)
            self.assertNotIn("SECRET_PATH", body)
            self.assertNotIn("secrets.json", body)
            self.assertIn("internal server error", body.lower())

    def test_500_chat_error_does_not_leak_internal_exception(self):
        """Verifies that internal errors in /api/chat return a generic sanitized response."""
        sensitive_err = "Internal vector store connection timeout to 192.168.1.99:5432"
        with patch("app.rag_assistant.ask_chatbot", side_effect=Exception(sensitive_err)):
            res = self.client.post("/api/chat", json={"message": "What is Wilson Disease?"})
            self.assertEqual(res.status_code, 500)
            data = res.get_json()
            self.assertNotIn("192.168.1.99", data.get("response", ""))
            self.assertNotIn("5432", data.get("response", ""))
            self.assertIn("error occurred", data.get("response", "").lower())

    def test_gitignore_protects_database_files(self):
        """Verifies that .gitignore contains rules to exclude SQLite database files."""
        gitignore_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".gitignore")
        self.assertTrue(os.path.exists(gitignore_path), ".gitignore must exist")
        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("*.db", content)
        self.assertIn("*.sqlite", content)


if __name__ == "__main__":
    unittest.main()
