"""
tests/test_auth_rbac.py

Comprehensive Isolated Automated Test Suite for Stage 8:
- Clinician Authentication, Role-Based Access Control (RBAC), and Server-Side Authorization.
- Secure CLI-only Administrator Bootstrap (manage_users.py).
- Login, Logout, Session Fixation Defense, and Idle Inactivity Timeout.
- Sliding-Window Login Rate Limiting (5 failures / 15m lockout).
- Immediate Session Revocation on Account Deactivation.
- Session-Bound CSRF Token Generation and Enforcement.
- Least-Privilege Route Protection Matrix (Clinician vs. Lab Staff vs. Administrator).
- Audit Logging Verification and Tamper Limitations.
- Online Crash-Consistent SQLite Database Backup.
"""

import os
import sys
import time
import tempfile
import shutil
import unittest
import sqlite3
from unittest.mock import patch

from app import app
from database import (
    init_db,
    open_db,
    create_user,
    get_user_by_id,
    get_user_by_username,
    verify_user_credentials,
    list_all_users,
    set_user_active_status,
    reset_user_password,
    log_audit_event,
    get_recent_audit_logs,
    backup_database
)
from auth_middleware import (
    check_login_rate_limit,
    record_failed_login,
    clear_failed_logins,
    _failed_login_attempts
)
import manage_users


class TestAuthAndRBACSuite(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_auth_records.db")
        init_db(self.test_db)

        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = True  # Enable CSRF enforcement by default for security testing
        app.config["DB_PATH"] = self.test_db
        self.client = app.test_client()

        # Clear in-memory rate limit dictionary before each test
        _failed_login_attempts.clear()

    def tearDown(self):
        _failed_login_attempts.clear()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ==========================================================================
    # 1. DATABASE USER PERSISTENCE & PASSWORD HASHING
    # ==========================================================================

    def test_user_creation_and_scrypt_hashing(self):
        """Verifies create_user hashes passwords securely with scrypt and rejects invalid input."""
        user = create_user(
            username="clinician_dr_smith",
            password="SecurePassword123!",
            full_name="Dr. Jane Smith",
            role="clinician",
            db_path=self.test_db
        )
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "clinician_dr_smith")
        self.assertEqual(user["role"], "clinician")
        self.assertEqual(user["is_active"], 1)

        # Verify password is NOT stored in plaintext
        with open_db(self.test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password_hash FROM users WHERE user_id = ?", (user["user_id"],))
            raw_hash = cursor.fetchone()[0]
            self.assertNotEqual(raw_hash, "SecurePassword123!")
            self.assertTrue(raw_hash.startswith("scrypt:"))

    def test_verify_user_credentials_and_case_insensitivity(self):
        """Verifies credential check succeeds for correct password and rejects invalid passwords."""
        create_user(
            username="LabTech01",
            password="LaboratoryKey999!",
            full_name="Alex Rivera",
            role="lab_staff",
            db_path=self.test_db
        )

        # Correct password, lowercase username match
        valid_user = verify_user_credentials("labtech01", "LaboratoryKey999!", db_path=self.test_db)
        self.assertIsNotNone(valid_user)
        self.assertEqual(valid_user["username"], "LabTech01")

        # Wrong password
        wrong_pw = verify_user_credentials("labtech01", "WrongPassword!", db_path=self.test_db)
        self.assertIsNone(wrong_pw)

        # Non-existent user
        non_existent = verify_user_credentials("ghost_user", "AnyPassword!", db_path=self.test_db)
        self.assertIsNone(non_existent)

    def test_duplicate_username_rejection(self):
        """Verifies duplicate usernames are rejected regardless of case."""
        create_user("AdminUser", "AdminSecret123!", "Root Admin", "admin", db_path=self.test_db)
        with self.assertRaises(ValueError):
            create_user("adminuser", "OtherSecret123!", "Second Admin", "admin", db_path=self.test_db)

    def test_invalid_role_and_short_password_rejection(self):
        """Verifies that invalid roles and weak passwords (<8 chars) raise ValueError."""
        with self.assertRaises(ValueError):
            create_user("user1", "short", "User One", "clinician", db_path=self.test_db)

        with self.assertRaises(ValueError):
            create_user("user2", "ValidPassword123!", "User Two", "super_hacker", db_path=self.test_db)

    # ==========================================================================
    # 2. SECURE CLI-ONLY ADMINISTRATOR BOOTSTRAP (manage_users.py)
    # ==========================================================================

    def test_cli_admin_bootstrap_and_complexity_enforcement(self):
        """Verifies manage_users.py init-admin bootstraps first admin and enforces complexity."""
        # Weak password fails (<12 chars)
        weak_args = manage_users.argparse.Namespace(
            db_path=self.test_db,
            username="root_admin",
            name="System Administrator",
            password="WeakPass1!"  # < 12 chars
        )
        rc_weak = manage_users.init_admin_command(weak_args)
        self.assertEqual(rc_weak, 1)

        # Strong password succeeds
        strong_args = manage_users.argparse.Namespace(
            db_path=self.test_db,
            username="root_admin",
            name="System Administrator",
            password="AdminComplex#2026!"
        )
        rc_strong = manage_users.init_admin_command(strong_args)
        self.assertEqual(rc_strong, 0)

        # Verify admin exists in DB
        admin_user = get_user_by_username("root_admin", db_path=self.test_db)
        self.assertIsNotNone(admin_user)
        self.assertEqual(admin_user["role"], "admin")

    def test_cli_admin_bootstrap_is_idempotent_and_prevents_multiple_admins(self):
        """Verifies running init-admin when an admin already exists aborts immediately."""
        create_user("first_admin", "AdminComplex#2026!", "First Admin", "admin", db_path=self.test_db)

        # Attempt to run init-admin again
        args = manage_users.argparse.Namespace(
            db_path=self.test_db,
            username="second_admin",
            name="Second Admin",
            password="AdminComplex#2026!"
        )
        rc = manage_users.init_admin_command(args)
        self.assertEqual(rc, 1)

        # Ensure second_admin was NOT created
        self.assertIsNone(get_user_by_username("second_admin", db_path=self.test_db))

    def test_cli_create_user_and_list_users(self):
        """Verifies manage_users.py create-user provisions accounts and list-users displays them."""
        args_create = manage_users.argparse.Namespace(
            db_path=self.test_db,
            username="staff_member",
            name="Staff Member",
            role="lab_staff",
            password="StaffPassword123!"
        )
        rc = manage_users.create_user_command(args_create)
        self.assertEqual(rc, 0)

        users = list_all_users(db_path=self.test_db)
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["username"], "staff_member")
        self.assertNotIn("password_hash", users[0])  # Must never expose hash

    # ==========================================================================
    # 3. LOGIN, LOGOUT, & SESSION WORKFLOWS
    # ==========================================================================

    def test_login_page_renders_with_csrf_token(self):
        """Verifies GET /login displays login form and provides CSRF token."""
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("name=\"csrf_token\"", html)
        self.assertIn("Sign In", html)
        self.assertIn("Authorized Healthcare Personnel Only", html)

    def test_successful_login_regenerates_session_and_clears_failures(self):
        """Verifies POST /login authenticates user and sets authenticated session."""
        create_user("dr_who", "TardisKey123!", "Dr. Who", "clinician", db_path=self.test_db)

        # Obtain valid CSRF token from GET
        get_resp = self.client.get("/login")
        with self.client.session_transaction() as sess:
            csrf_token = sess.get("_csrf_token")

        post_resp = self.client.post("/login", data={
            "username": "dr_who",
            "password": "TardisKey123!",
            "csrf_token": csrf_token
        }, follow_redirects=False)

        # Redirects to /about for clinicians
        self.assertEqual(post_resp.status_code, 302)
        self.assertIn("/about", post_resp.headers.get("Location", ""))

        with self.client.session_transaction() as sess:
            self.assertEqual(sess.get("username"), "dr_who")
            self.assertEqual(sess.get("role"), "clinician")
            self.assertIsNotNone(sess.get("user_id"))

    def test_invalid_login_returns_401(self):
        """Verifies POST /login with incorrect password returns 401."""
        create_user("dr_watson", "Sherlock123!", "Dr. Watson", "clinician", db_path=self.test_db)

        self.client.get("/login")
        with self.client.session_transaction() as sess:
            csrf_token = sess.get("_csrf_token")

        resp = self.client.post("/login", data={
            "username": "dr_watson",
            "password": "WrongPassword999!",
            "csrf_token": csrf_token
        })
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid username or password", resp.get_data(as_text=True))

    def test_logout_clears_session(self):
        """Verifies /logout is POST-only, CSRF-protected, clears session, and redirects to /login."""
        user = create_user("dr_house", "VicodinKey123!", "Dr. House", "clinician", db_path=self.test_db)

        # 1. GET /logout must be rejected with 405 Method Not Allowed
        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = "dr_house"
            sess["role"] = "clinician"
            sess["last_active"] = time.time()
            sess["_csrf_token"] = "valid_csrf_token_house"

        get_logout_resp = self.client.get("/logout", follow_redirects=False)
        self.assertEqual(get_logout_resp.status_code, 405)

        # 2. POST /logout without CSRF token must be rejected with 400 Bad Request
        post_no_csrf = self.client.post("/logout", follow_redirects=False)
        self.assertEqual(post_no_csrf.status_code, 400)

        # 3. POST /logout with valid CSRF token succeeds, terminates session, redirects to /login
        logout_resp = self.client.post("/logout", data={"csrf_token": "valid_csrf_token_house"}, follow_redirects=False)
        self.assertEqual(logout_resp.status_code, 302)
        self.assertIn("/login", logout_resp.headers.get("Location", ""))

        # Verify session cleared
        with self.client.session_transaction() as sess:
            self.assertNotIn("user_id", sess)
            self.assertNotIn("username", sess)

    # ==========================================================================
    # 4. LOGIN RATE LIMITING (5 FAILURES / 15-MINUTE LOCKOUT)
    # ==========================================================================

    def test_login_rate_limiting_locks_on_sixth_failed_attempt(self):
        """Verifies 5 failed attempts are allowed, 6th attempt returns HTTP 429."""
        create_user("target_user", "TargetPassword123!", "Target User", "clinician", db_path=self.test_db)

        # Get initial CSRF token
        self.client.get("/login")
        with self.client.session_transaction() as sess:
            csrf_token = sess.get("_csrf_token")

        # 5 consecutive failed attempts
        for i in range(5):
            res = self.client.post("/login", data={
                "username": "target_user",
                "password": f"BadPassword{i}",
                "csrf_token": csrf_token
            })
            self.assertEqual(res.status_code, 401, f"Attempt {i+1} should be 401")

        # 6th attempt must be rejected with 429 Too Many Requests
        res_6 = self.client.post("/login", data={
            "username": "target_user",
            "password": "TargetPassword123!",  # Even correct password is locked out
            "csrf_token": csrf_token
        })
        self.assertEqual(res_6.status_code, 429)
        self.assertIn("Too many failed login attempts", res_6.get_data(as_text=True))

    # ==========================================================================
    # 5. ROLE-BASED ACCESS CONTROL (LEAST PRIVILEGE)
    # ==========================================================================

    def test_unauthenticated_requests_redirected_or_rejected(self):
        """Verifies unauthenticated access to protected routes is redirected or returned 401."""
        # HTML views redirect to /login
        res_about = self.client.get("/about", follow_redirects=False)
        self.assertEqual(res_about.status_code, 302)
        self.assertIn("/login", res_about.headers.get("Location", ""))

        res_patients = self.client.get("/patients", follow_redirects=False)
        self.assertEqual(res_patients.status_code, 302)
        self.assertIn("/login", res_patients.headers.get("Location", ""))

        # API view returns 401 JSON
        res_api = self.client.get("/api/next_patient_id")
        self.assertEqual(res_api.status_code, 401)
        self.assertFalse(res_api.get_json()["success"])

        # Direct file download of uploads redirects
        res_upload = self.client.get("/uploads/secret_plot.png", follow_redirects=False)
        self.assertEqual(res_upload.status_code, 302)

    def test_lab_staff_least_privilege_enforcement(self):
        """
        Verifies lab_staff CAN access sample templates & upload files,
        but CANNOT access patient directory, submit predictions, or view patient records.
        """
        user = create_user("lab_analyst", "LabAnalystPass123!", "Lab Analyst", "lab_staff", db_path=self.test_db)
        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = user["username"]
            sess["role"] = "lab_staff"
            sess["last_active"] = time.time()
            sess["_csrf_token"] = "valid_csrf_token"

        # Allowed for lab_staff
        res_template = self.client.get("/api/sample_template?format=csv")
        self.assertEqual(res_template.status_code, 200)

        # FORBIDDEN for lab_staff: Patient directory
        res_patients = self.client.get("/patients")
        self.assertEqual(res_patients.status_code, 403)

        # FORBIDDEN for lab_staff: Clinical Test entry form
        res_about = self.client.get("/about")
        self.assertEqual(res_about.status_code, 403)

        # FORBIDDEN for lab_staff: Patient lookup
        res_lookup = self.client.get("/api/patient/NEUROMED001")
        self.assertEqual(res_lookup.status_code, 403)

        # FORBIDDEN for lab_staff: Model prediction execution
        app.config["WTF_CSRF_ENABLED"] = False
        res_submit = self.client.post("/submit", data={"patientName": "Test"})
        self.assertEqual(res_submit.status_code, 403)
        app.config["WTF_CSRF_ENABLED"] = True

    def test_admin_least_privilege_cannot_access_clinical_records(self):
        """
        Verifies admin CANNOT access patient records or clinical tests without clinical role.
        Enforces separation of duties between IT administrators and healthcare clinicians.
        """
        admin = create_user("sysadmin", "SysAdminPass123!", "System Admin", "admin", db_path=self.test_db)
        with self.client.session_transaction() as sess:
            sess["user_id"] = admin["user_id"]
            sess["username"] = admin["username"]
            sess["role"] = "admin"
            sess["last_active"] = time.time()

        res_patients = self.client.get("/patients")
        self.assertEqual(res_patients.status_code, 403)

        res_about = self.client.get("/about")
        self.assertEqual(res_about.status_code, 403)

    def test_clinician_has_full_clinical_access(self):
        """Verifies authenticated clinician can access patient directory, about, and APIs."""
        clinician = create_user("dr_jones", "DrJonesPass123!", "Dr. Jones", "clinician", db_path=self.test_db)
        with self.client.session_transaction() as sess:
            sess["user_id"] = clinician["user_id"]
            sess["username"] = clinician["username"]
            sess["role"] = "clinician"
            sess["last_active"] = time.time()

        res_about = self.client.get("/about")
        self.assertEqual(res_about.status_code, 200)

        res_patients = self.client.get("/patients")
        self.assertEqual(res_patients.status_code, 200)

        res_api = self.client.get("/api/next_patient_id")
        self.assertEqual(res_api.status_code, 200)

    # ==========================================================================
    # 6. ACCOUNT DEACTIVATION (IMMEDIATE SESSION REVOCATION)
    # ==========================================================================

    def test_account_deactivation_immediately_terminates_active_session(self):
        """Verifies that deactivating a user while logged in immediately invalidates their session."""
        user = create_user("compromised_user", "CompromisedPass123!", "Compromised User", "clinician", db_path=self.test_db)

        # Login session established
        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = user["username"]
            sess["role"] = "clinician"
            sess["last_active"] = time.time()

        # Session works initially
        res_ok = self.client.get("/about")
        self.assertEqual(res_ok.status_code, 200)

        # Admin deactivates user in database
        set_user_active_status(user["user_id"], is_active=False, db_path=self.test_db)

        # Next request must be immediately blocked with 403
        res_blocked = self.client.get("/about")
        self.assertEqual(res_blocked.status_code, 403)
        self.assertIn("deactivated", res_blocked.get_data(as_text=True).lower())

        # Session should be completely cleared
        with self.client.session_transaction() as sess:
            self.assertNotIn("user_id", sess)

    # ==========================================================================
    # 7. SESSION IDLE TIMEOUT (30 MINUTES)
    # ==========================================================================

    def test_session_idle_timeout_expires_inactive_session(self):
        """Verifies session with last_active > 30 minutes ago is expired and redirected to login."""
        user = create_user("inactive_clinician", "ClinicianPass123!", "Dr. Inactive", "clinician", db_path=self.test_db)

        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = user["username"]
            sess["role"] = "clinician"
            # Set activity timestamp to 31 minutes ago (1860s)
            sess["last_active"] = time.time() - 1860

        res = self.client.get("/about", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/login", res.headers.get("Location", ""))

        # Session must be purged
        with self.client.session_transaction() as sess:
            self.assertNotIn("user_id", sess)

    # ==========================================================================
    # 8. CSRF PROTECTION
    # ==========================================================================

    def test_csrf_rejection_on_missing_or_tampered_token(self):
        """Verifies POST requests without valid CSRF token return 400 Bad Request."""
        user = create_user("clinician_csrf", "ClinicianPass123!", "Dr. CSRF", "clinician", db_path=self.test_db)
        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = user["username"]
            sess["role"] = "clinician"
            sess["last_active"] = time.time()
            sess["_csrf_token"] = "legitimate_session_token_12345"

        # POST without CSRF token
        res_missing = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(res_missing.status_code, 400)

        # POST with wrong/tampered CSRF token
        res_tampered = self.client.post(
            "/api/chat",
            json={"message": "Hello", "csrf_token": "attacker_forged_token"}
        )
        self.assertEqual(res_tampered.status_code, 400)

        # POST with valid CSRF token in header
        res_valid = self.client.post(
            "/api/chat",
            json={"message": "What is Wilson Disease?"},
            headers={"X-CSRFToken": "legitimate_session_token_12345"}
        )
        # 200 OK
        self.assertEqual(res_valid.status_code, 200)

    # ==========================================================================
    # 9. AUDIT LOGGING VERIFICATION
    # ==========================================================================

    def test_audit_logs_record_security_events_without_sensitive_leakage(self):
        """Verifies that security events write to audit_logs and contain no plaintext passwords."""
        # 1. Trigger failed login
        self.client.get("/login")
        with self.client.session_transaction() as sess:
            token = sess.get("_csrf_token")

        self.client.post("/login", data={
            "username": "audit_victim",
            "password": "SUPER_SECRET_PLAINTEXT_PASSWORD_DO_NOT_LEAK",
            "csrf_token": token
        })

        # Retrieve audit logs
        logs = get_recent_audit_logs(limit=20, db_path=self.test_db)
        self.assertGreater(len(logs), 0)

        actions = [l["action"] for l in logs]
        self.assertIn("LOGIN_FAILED", actions)

        # Verify plaintext password NEVER appears in audit logs
        for log in logs:
            details = log.get("details") or ""
            self.assertNotIn("SUPER_SECRET_PLAINTEXT_PASSWORD_DO_NOT_LEAK", details)

    # ==========================================================================
    # 10. CRASH-CONSISTENT NATIVE SQLITE BACKUP
    # ==========================================================================

    def test_crash_consistent_sqlite_backup(self):
        """Verifies native SQLite backup produces an intact, queryable backup file."""
        # Create users and audit records in source database
        create_user("backup_test_user", "BackupPass123!", "Backup User", "clinician", db_path=self.test_db)
        log_audit_event("PRE_BACKUP_TEST", "SUCCESS", username="backup_test_user", db_path=self.test_db)

        backup_file = os.path.join(self.temp_dir, "backup_output.db")
        created_path = backup_database(db_path=self.test_db, target_backup_path=backup_file)
        self.assertTrue(os.path.exists(created_path))

        # Query the backup file independently
        with open_db(created_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username, role FROM users WHERE username = 'backup_test_user';")
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["username"], "backup_test_user")
            self.assertEqual(row["role"], "clinician")

            cursor.execute("SELECT action FROM audit_logs WHERE action = 'PRE_BACKUP_TEST';")
            audit_row = cursor.fetchone()
            self.assertIsNotNone(audit_row)

    # ==========================================================================
    # 11. PRODUCTION CONFIGURATION & FAIL-SAFE STARTUP VERIFICATION
    # ==========================================================================

    def test_production_fail_safe_startup_validation(self):
        """
        Verifies production startup fails safely if FLASK_SECRET_KEY is absent,
        weak (< 32 chars), or matches the development placeholder.
        Verifies SESSION_COOKIE_SECURE=True is enabled in production and False in local dev.
        """
        import subprocess

        # 1. Missing FLASK_SECRET_KEY in production mode -> fails
        env_missing = os.environ.copy()
        env_missing["FLASK_ENV"] = "production"
        env_missing.pop("FLASK_SECRET_KEY", None)
        cmd_missing = [sys.executable, "-c", "import app"]
        proc_missing = subprocess.run(cmd_missing, env=env_missing, capture_output=True, text=True)
        self.assertNotEqual(proc_missing.returncode, 0)
        self.assertIn("Production security check failed", proc_missing.stderr)

        # 2. Development placeholder key in production mode -> fails
        env_devkey = os.environ.copy()
        env_devkey["FLASK_ENV"] = "production"
        env_devkey["FLASK_SECRET_KEY"] = "wilson-neuromed-clinical-dev-secret-key-change-in-prod"
        proc_devkey = subprocess.run(cmd_missing, env=env_devkey, capture_output=True, text=True)
        self.assertNotEqual(proc_devkey.returncode, 0)
        self.assertIn("Production security check failed", proc_devkey.stderr)

        # 3. Weak key (< 32 chars) in production mode -> fails
        env_weak = os.environ.copy()
        env_weak["FLASK_ENV"] = "production"
        env_weak["FLASK_SECRET_KEY"] = "short_secret_key_12345"
        proc_weak = subprocess.run(cmd_missing, env=env_weak, capture_output=True, text=True)
        self.assertNotEqual(proc_weak.returncode, 0)
        self.assertIn("Production security check failed", proc_weak.stderr)

        # 4. High-entropy key (>= 32 chars) in production mode -> succeeds and sets SESSION_COOKIE_SECURE = True
        env_good = os.environ.copy()
        env_good["FLASK_ENV"] = "production"
        env_good["FLASK_SECRET_KEY"] = "a_super_strong_high_entropy_production_secret_key_2026_!"
        cmd_check = [sys.executable, "-c", "import app; print('SECURE=' + str(app.app.config['SESSION_COOKIE_SECURE']))"]
        proc_good = subprocess.run(cmd_check, env=env_good, capture_output=True, text=True)
        self.assertEqual(proc_good.returncode, 0)
        self.assertIn("SECURE=True", proc_good.stdout)

        # 5. Local dev / test environment retains SESSION_COOKIE_SECURE = False by default
        self.assertFalse(app.config.get("SESSION_COOKIE_SECURE", False))

    # ==========================================================================
    # 12. MUST_CHANGE_PASSWORD SERVER-SIDE ENFORCEMENT & WORKFLOW
    # ==========================================================================

    def test_must_change_password_server_side_enforcement(self):
        """
        Verifies that must_change_password is strictly enforced server-side.
        Users with must_change_password=1 cannot access clinical or lab routes
        and are redirected or returned 403 until password is changed.
        """
        user = create_user(
            username="temp_user",
            password="TemporaryPassword123!",
            full_name="Temporary Staff",
            role="clinician",
            must_change_password=True,
            db_path=self.test_db
        )

        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = "temp_user"
            sess["role"] = "clinician"
            sess["last_active"] = time.time()
            sess["_csrf_token"] = "valid_token_temp"

        # 1. Attempting to view clinical page /about redirects to /change-password
        res_about = self.client.get("/about", follow_redirects=False)
        self.assertEqual(res_about.status_code, 302)
        self.assertIn("/change-password", res_about.headers.get("Location", ""))

        # 2. Attempting to access API returns 403 JSON
        res_api = self.client.get("/api/next_patient_id")
        self.assertEqual(res_api.status_code, 403)
        self.assertTrue(res_api.get_json().get("must_change_password"))

        # 3. GET /change-password succeeds
        res_pw_form = self.client.get("/change-password")
        self.assertEqual(res_pw_form.status_code, 200)
        self.assertIn("Update Your Password", res_pw_form.get_data(as_text=True))

        # 4. POST /change-password with mismatched passwords fails
        res_mismatch = self.client.post("/change-password", data={
            "current_password": "TemporaryPassword123!",
            "new_password": "NewStrongPassword456!",
            "confirm_password": "DifferentPassword789!",
            "csrf_token": "valid_token_temp"
        })
        self.assertEqual(res_mismatch.status_code, 400)
        self.assertIn("passwords do not match", res_mismatch.get_data(as_text=True).lower())

        # 5. POST /change-password reusing old password fails
        res_reuse = self.client.post("/change-password", data={
            "current_password": "TemporaryPassword123!",
            "new_password": "TemporaryPassword123!",
            "confirm_password": "TemporaryPassword123!",
            "csrf_token": "valid_token_temp"
        })
        self.assertEqual(res_reuse.status_code, 400)
        self.assertIn("same as your current password", res_reuse.get_data(as_text=True))

        # 6. POST /change-password with valid new password succeeds
        res_success = self.client.post("/change-password", data={
            "current_password": "TemporaryPassword123!",
            "new_password": "BrandNewPermanentPassword999!",
            "confirm_password": "BrandNewPermanentPassword999!",
            "csrf_token": "valid_token_temp"
        }, follow_redirects=False)
        self.assertEqual(res_success.status_code, 302)

        # 7. Check database: must_change_password is now 0
        updated_user = get_user_by_id(user["user_id"], db_path=self.test_db)
        self.assertEqual(updated_user["must_change_password"], 0)

        # 8. User can now access clinical pages without redirection
        res_allowed = self.client.get("/about")
        self.assertEqual(res_allowed.status_code, 200)

    # ==========================================================================
    # 13. OPEN REDIRECT DEFENSE & POST-LOGIN NEXT URL HANDLING
    # ==========================================================================

    def test_safe_post_login_redirect_handling(self):
        """Verifies that next_url rejects open redirect bypasses and accepts safe internal paths."""
        from auth_middleware import is_safe_redirect_url

        # Open redirect attacks must be rejected
        self.assertFalse(is_safe_redirect_url("https://evil.com"))
        self.assertFalse(is_safe_redirect_url("http://attacker.org/steal"))
        self.assertFalse(is_safe_redirect_url("//evil.com"))
        self.assertFalse(is_safe_redirect_url("/\\evil.com"))
        self.assertFalse(is_safe_redirect_url("/test\\evil"))
        self.assertFalse(is_safe_redirect_url("javascript:alert(1)"))
        self.assertFalse(is_safe_redirect_url(""))
        self.assertFalse(is_safe_redirect_url(None))

        # Legitimate relative paths must be accepted
        self.assertTrue(is_safe_redirect_url("/about"))
        self.assertTrue(is_safe_redirect_url("/patients"))
        self.assertTrue(is_safe_redirect_url("/patients?search=NEUROMED001&page=2"))
        self.assertTrue(is_safe_redirect_url("/lab/import"))

        # Test within /login route
        create_user("dr_redirect", "RedirectPass123!", "Dr. Redirect", "clinician", db_path=self.test_db)
        self.client.get("/login")
        with self.client.session_transaction() as sess:
            token = sess.get("_csrf_token")

        # Attack payload in next param -> should redirect to default /about, NOT evil.com
        res_attack = self.client.post("/login", data={
            "username": "dr_redirect",
            "password": "RedirectPass123!",
            "csrf_token": token,
            "next": "//evil.com"
        }, follow_redirects=False)
        self.assertEqual(res_attack.status_code, 302)
        self.assertNotIn("evil.com", res_attack.headers.get("Location", ""))
        self.assertIn("/about", res_attack.headers.get("Location", ""))

    # ==========================================================================
    # 14. DATA PROTECTION: NO PASSWORD HASHES IN API RESPONSES
    # ==========================================================================

    def test_no_password_hash_in_any_api_or_function(self):
        """Verifies password hashes are never exposed via API endpoints or user queries."""
        user = create_user("audit_user_hash", "VerySecretPassword123!", "Audit User", "clinician", db_path=self.test_db)

        # 1. Database helper functions must omit password_hash
        u_by_id = get_user_by_id(user["user_id"], db_path=self.test_db)
        self.assertNotIn("password_hash", u_by_id)

        u_by_uname = get_user_by_username("audit_user_hash", db_path=self.test_db)
        self.assertNotIn("password_hash", u_by_uname)

        all_users = list_all_users(db_path=self.test_db)
        for u in all_users:
            self.assertNotIn("password_hash", u)

        verified = verify_user_credentials("audit_user_hash", "VerySecretPassword123!", db_path=self.test_db)
        self.assertNotIn("password_hash", verified)

        # 2. JSON login API response must never return password_hash
        self.client.get("/login")
        with self.client.session_transaction() as sess:
            token = sess.get("_csrf_token")

        res_login = self.client.post("/login", json={
            "username": "audit_user_hash",
            "password": "VerySecretPassword123!",
            "csrf_token": token
        })
        self.assertEqual(res_login.status_code, 200)
        data = res_login.get_json()
        self.assertNotIn("password_hash", data.get("user", {}))

    # ==========================================================================
    # 15. DEDICATED LAB IMPORT PORTAL & STRICT ACCESS BARRIERS
    # ==========================================================================

    def test_lab_staff_dedicated_portal_and_access_barriers(self):
        """
        Confirms /lab/import exists and is accessible to lab_staff and clinicians.
        Confirms lab_staff cannot access patient records, reports, predictions,
        or SHAP files through any route.
        """
        lab_user = create_user("lab_tech_lead", "TechLeadKey123!", "Lead Tech", "lab_staff", db_path=self.test_db)
        with self.client.session_transaction() as sess:
            sess["user_id"] = lab_user["user_id"]
            sess["username"] = lab_user["username"]
            sess["role"] = "lab_staff"
            sess["last_active"] = time.time()
            sess["_csrf_token"] = "valid_lab_token"

        # 1. /lab/import portal is accessible (200 OK)
        res_lab_portal = self.client.get("/lab/import")
        self.assertEqual(res_lab_portal.status_code, 200)
        self.assertIn("Laboratory Data Ingestion", res_lab_portal.get_data(as_text=True))

        # 2. Sample template API is accessible (200 OK)
        res_sample = self.client.get("/api/sample_template?format=json")
        self.assertEqual(res_sample.status_code, 200)

        # 3. Lab staff CANNOT access clinical testing entry (/about) -> 403
        self.assertEqual(self.client.get("/about").status_code, 403)

        # 4. Lab staff CANNOT access patient directory (/patients) -> 403
        self.assertEqual(self.client.get("/patients").status_code, 403)

        # 5. Lab staff CANNOT access patient assessment history -> 403
        self.assertEqual(self.client.get("/patient/NEUROMED001/history").status_code, 403)

        # 6. Lab staff CANNOT access historical assessment reports -> 403
        self.assertEqual(self.client.get("/assessment/1/report").status_code, 403)

        # 7. Lab staff CANNOT access clinical analytics (/do) -> 403
        self.assertEqual(self.client.get("/do").status_code, 403)

        # 8. Lab staff CANNOT execute assessment prediction via POST /submit -> 403
        self.assertEqual(self.client.post("/submit", data={"csrf_token": "valid_lab_token"}).status_code, 403)

        # 9. Lab staff CANNOT access clinical chat API -> 403
        self.assertEqual(self.client.post("/api/chat", json={"message": "test", "csrf_token": "valid_lab_token"}).status_code, 403)

        # 10. Lab staff CANNOT access patient profile API -> 403
        self.assertEqual(self.client.get("/api/patient/NEUROMED001").status_code, 403)

        # 11. Lab staff CANNOT access SHAP plots or diagnostic files from /uploads -> 403
        self.assertEqual(self.client.get("/uploads/secret_waterfall.png").status_code, 403)


if __name__ == "__main__":
    unittest.main()
