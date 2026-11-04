"""
tests/test_admin_dashboard.py

Comprehensive Isolated Automated Test Suite for Chief Doctor Admin Dashboard & Governance:
1. Chief Doctor Dedicated Login (/admin/login) vs Staff Portal (/login).
2. Strict RBAC Enforcement: Clinicians and Lab Staff cannot access Admin Dashboard or Admin APIs (HTTP 403).
3. Chief Doctor Admin Dashboard (/admin/dashboard) full access and data aggregation.
4. User Management APIs (/api/admin/users): listing, creation, activation/deactivation, password reset.
5. Self-Deactivation Prevention: Active Chief Doctor cannot deactivate their own account.
6. Audit Logging & Export APIs (/api/admin/audit_logs and /api/admin/audit_logs/export).
7. System Health & Infrastructure Telemetry API (/api/admin/system_health).
8. Model Governance API (/api/admin/model_governance).
9. Full Clinician Capabilities accessible by Chief Doctor.
"""

import os
import sys
import tempfile
import unittest
import json

from app import app
from database import (
    init_db,
    create_user,
    get_user_by_id,
    list_all_users,
    log_audit_event,
    get_recent_audit_logs,
    get_system_health_stats,
    get_admin_overview_metrics,
    create_or_update_patient
)


class TestChiefDoctorAdminDashboard(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_admin_records.db")
        init_db(self.test_db)

        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["DB_PATH"] = self.test_db
        self.client = app.test_client()

        # Seed standard users
        self.admin = create_user("chief_doc", "ChiefPass123!@", "Dr. Chief Administrator", "admin", db_path=self.test_db)
        self.clinician = create_user("dr_jones", "ClinicianPass123!@", "Dr. Indiana Jones", "clinician", db_path=self.test_db)
        self.lab_staff = create_user("tech_smith", "LabPass123!@", "Technician Smith", "lab_staff", db_path=self.test_db)

    def tear_down(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _login_as(self, username, password):
        """Helper to establish an authenticated session."""
        return self.client.post("/login", data={
            "username": username,
            "password": password
        }, follow_redirects=False)

    # --------------------------------------------------------------------------
    # 1. ADMIN LOGIN & AUTHENTICATION TESTS
    # --------------------------------------------------------------------------

    def test_admin_login_get_page(self):
        """Verifies GET /admin/login returns 200 and renders Chief Doctor portal."""
        res = self.client.get("/admin/login")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Chief Doctor Portal", res.get_data(as_text=True))
        self.assertIn("Executive Governance Access", res.get_data(as_text=True))

    def test_admin_login_post_success(self):
        """Verifies Chief Doctor can successfully log in via /admin/login."""
        res = self.client.post("/admin/login", data={
            "username": "chief_doc",
            "password": "ChiefPass123!@"
        }, follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/admin/dashboard", res.headers.get("Location", ""))

    def test_admin_login_rejects_clinician(self):
        """Verifies non-admin clinician credentials cannot authenticate via /admin/login."""
        res = self.client.post("/admin/login", data={
            "username": "dr_jones",
            "password": "ClinicianPass123!@"
        }, follow_redirects=False)
        self.assertEqual(res.status_code, 403)
        self.assertIn("Access restricted: This portal is exclusively for the Chief Doctor", res.get_data(as_text=True))

    def test_admin_login_rejects_bad_password(self):
        """Verifies invalid password fails authentication on /admin/login."""
        res = self.client.post("/admin/login", data={
            "username": "chief_doc",
            "password": "WrongPassword999!"
        }, follow_redirects=False)
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid Chief Doctor administrator username or password", res.get_data(as_text=True))

    # --------------------------------------------------------------------------
    # 2. RBAC ACCESS CONTROL TESTS
    # --------------------------------------------------------------------------

    def test_unauthenticated_access_denied(self):
        """Verifies unauthenticated user cannot access /admin/dashboard."""
        res = self.client.get("/admin/dashboard")
        self.assertEqual(res.status_code, 302)
        self.assertIn("/login", res.headers.get("Location", ""))

    def test_clinician_cannot_access_admin_dashboard(self):
        """Verifies clinician receives HTTP 403 Forbidden when accessing /admin/dashboard."""
        self._login_as("dr_jones", "ClinicianPass123!@")
        res = self.client.get("/admin/dashboard")
        self.assertEqual(res.status_code, 403)

    def test_lab_staff_cannot_access_admin_dashboard(self):
        """Verifies lab staff receives HTTP 403 Forbidden when accessing /admin/dashboard."""
        self._login_as("tech_smith", "LabPass123!@")
        res = self.client.get("/admin/dashboard")
        self.assertEqual(res.status_code, 403)

    def test_admin_can_access_admin_dashboard(self):
        """Verifies Chief Doctor can access /admin/dashboard successfully."""
        self._login_as("chief_doc", "ChiefPass123!@")
        res = self.client.get("/admin/dashboard")
        self.assertEqual(res.status_code, 200)
        content = res.get_data(as_text=True)
        self.assertIn("Chief Doctor Executive Command Center", content)
        self.assertIn("User &amp; Role Management", content)
        self.assertIn("Clinical AI Model Governance", content)

    # --------------------------------------------------------------------------
    # 3. USER MANAGEMENT API TESTS
    # --------------------------------------------------------------------------

    def test_admin_can_list_users(self):
        """Verifies GET /api/admin/users lists all registered users."""
        self._login_as("chief_doc", "ChiefPass123!@")
        res = self.client.get("/api/admin/users")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        usernames = [u["username"] for u in data["users"]]
        self.assertIn("chief_doc", usernames)
        self.assertIn("dr_jones", usernames)

    def test_clinician_cannot_list_users(self):
        """Verifies clinician receives HTTP 403 on GET /api/admin/users."""
        self._login_as("dr_jones", "ClinicianPass123!@")
        res = self.client.get("/api/admin/users")
        self.assertEqual(res.status_code, 403)

    def test_admin_can_create_user(self):
        """Verifies Chief Doctor can provision a new clinician account."""
        self._login_as("chief_doc", "ChiefPass123!@")
        res = self.client.post("/api/admin/users", json={
            "username": "dr_newbie",
            "password": "StrongPassword999!",
            "full_name": "Dr. Newbie Clinician",
            "role": "clinician"
        })
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["user"]["username"], "dr_newbie")

    def test_admin_can_toggle_user_status(self):
        """Verifies Chief Doctor can deactivate and reactivate accounts."""
        self._login_as("chief_doc", "ChiefPass123!@")
        # Deactivate dr_jones
        res_deact = self.client.post(f"/api/admin/users/{self.clinician['user_id']}/toggle_status", json={
            "is_active": False
        })
        self.assertEqual(res_deact.status_code, 200)
        user_after = get_user_by_id(self.clinician["user_id"], db_path=self.test_db)
        self.assertEqual(user_after["is_active"], 0)

        # Reactivate dr_jones
        res_act = self.client.post(f"/api/admin/users/{self.clinician['user_id']}/toggle_status", json={
            "is_active": True
        })
        self.assertEqual(res_act.status_code, 200)
        user_reactivated = get_user_by_id(self.clinician["user_id"], db_path=self.test_db)
        self.assertEqual(user_reactivated["is_active"], 1)

    def test_admin_cannot_deactivate_self(self):
        """Verifies Chief Doctor cannot deactivate their own active account."""
        self._login_as("chief_doc", "ChiefPass123!@")
        res = self.client.post(f"/api/admin/users/{self.admin['user_id']}/toggle_status", json={
            "is_active": False
        })
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Security policy prohibits self-deactivation", data["error"])

    def test_admin_can_reset_user_password(self):
        """Verifies Chief Doctor can reset user password."""
        self._login_as("chief_doc", "ChiefPass123!@")
        res = self.client.post(f"/api/admin/users/{self.clinician['user_id']}/reset_password", json={
            "new_password": "NewResetPassword123!"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["temporary_password"], "NewResetPassword123!")

    # --------------------------------------------------------------------------
    # 4. AUDIT LOGGING & EXPORT TESTS
    # --------------------------------------------------------------------------

    def test_admin_can_view_and_filter_audit_logs(self):
        """Verifies GET /api/admin/audit_logs returns paginated audit records."""
        log_audit_event(action="TEST_ACTION_XYZ", status="SUCCESS", username="chief_doc", details="Test audit", db_path=self.test_db)
        self._login_as("chief_doc", "ChiefPass123!@")
        res = self.client.get("/api/admin/audit_logs?action=TEST_ACTION_XYZ")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["total"], 1)
        self.assertEqual(data["logs"][0]["action"], "TEST_ACTION_XYZ")

    def test_admin_can_export_audit_logs_csv(self):
        """Verifies GET /api/admin/audit_logs/export returns CSV file."""
        self._login_as("chief_doc", "ChiefPass123!@")
        res = self.client.get("/api/admin/audit_logs/export")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/csv")
        csv_text = res.get_data(as_text=True)
        self.assertIn("Log ID,Timestamp (UTC),Actor Username,Action,Status,IP Address,Resource ID,Details", csv_text)

    # --------------------------------------------------------------------------
    # 5. SYSTEM HEALTH & MODEL GOVERNANCE TESTS
    # --------------------------------------------------------------------------

    def test_admin_system_health_api(self):
        """Verifies GET /api/admin/system_health returns valid system telemetry."""
        self._login_as("chief_doc", "ChiefPass123!@")
        res = self.client.get("/api/admin/system_health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["health"]["status"], "HEALTHY")
        self.assertIn("table_counts", data["health"])

    def test_admin_model_governance_api(self):
        """Verifies GET /api/admin/model_governance returns frozen policy status."""
        self._login_as("chief_doc", "ChiefPass123!@")
        res = self.client.get("/api/admin/model_governance")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["governance"]["governance_policy"]["status"], "FROZEN_IN_SERVICE")

    # --------------------------------------------------------------------------
    # 6. FULL CLINICIAN CAPABILITIES AVAILABLE TO ADMIN
    # --------------------------------------------------------------------------

    def test_admin_has_full_clinician_access(self):
        """Verifies Chief Doctor account has full access to all standard clinician routes."""
        self._login_as("chief_doc", "ChiefPass123!@")
        
        # 1. Clinical Test Page
        res_about = self.client.get("/about")
        self.assertEqual(res_about.status_code, 200)

        # 2. Patient Directory
        res_patients = self.client.get("/patients")
        self.assertEqual(res_patients.status_code, 200)

        # 3. Next Patient ID API
        res_pid = self.client.get("/api/next_patient_id")
        self.assertEqual(res_pid.status_code, 200)
        self.assertTrue(res_pid.get_json()["success"])

        # 4. RAG Monitor Page
        res_rag = self.client.get("/rag/monitor")
        self.assertEqual(res_rag.status_code, 200)


if __name__ == "__main__":
    unittest.main()

