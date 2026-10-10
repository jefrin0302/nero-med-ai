"""
auth_middleware.py

Authentication, Role-Based Access Control (RBAC), CSRF, and Session Hardening Middleware
for NeuroMed AI.
Provides:
1. Server-side @login_required and @roles_required decorators with least-privilege enforcement.
2. In-request active account verification against database (immediate session termination upon deactivation).
3. Sliding-window login rate limiting per IP and username (5 failures / 15-minute lockout).
4. Session-bound CSRF token issuance and validation.
5. Inactivity idle session timeout (30 minutes).
"""

import time
import secrets
import hmac
from functools import wraps
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional, List, Tuple, Dict

from flask import request, session, redirect, url_for, jsonify, render_template, abort, current_app

from database import (
    get_user_by_id,
    log_audit_event,
    DEFAULT_DB_PATH
)


def is_safe_redirect_url(target: Optional[str]) -> bool:
    """
    Validates that a post-login redirect URL is strictly an internal relative path.
    Prevents open redirect attacks (rejects external domains, schemes, //, /\\, backslashes).
    """
    if not target or not isinstance(target, str):
        return False
    clean = target.strip()
    if clean.startswith("//") or clean.startswith("/\\") or "\\" in clean:
        return False
    parsed = urlparse(clean)
    return parsed.scheme == "" and parsed.netloc == "" and clean.startswith("/")




# Inactivity timeout: 30 minutes (1800 seconds)
IDLE_TIMEOUT_SECONDS = 1800

# Rate limiting: max 5 failed attempts per 15 minutes (900 seconds)
RATE_LIMIT_WINDOW_SECONDS = 900
MAX_FAILED_ATTEMPTS = 5

# In-memory tracking of failed login attempts: (ip, username) -> list of timestamps
_failed_login_attempts: Dict[Tuple[str, str], List[float]] = {}


def get_client_ip() -> str:
    """Extracts client IP address safely."""
    if request.headers.get("X-Forwarded-For"):
        return request.headers.get("X-Forwarded-For").split(",")[0].strip()
    return request.remote_addr or "127.0.0.1"


# ==============================================================================
# LOGIN RATE LIMITING
# ==============================================================================

def check_login_rate_limit(ip: str, username: str) -> Tuple[bool, int]:
    """
    Checks whether the (ip, username) pair is locked due to too many failed attempts.
    Returns (is_allowed, remaining_seconds_locked).
    """
    now = time.time()
    key = (ip, (username or "").strip().lower())
    attempts = _failed_login_attempts.get(key, [])

    # Filter out attempts older than the sliding window
    valid_attempts = [t for t in attempts if (now - t) < RATE_LIMIT_WINDOW_SECONDS]
    _failed_login_attempts[key] = valid_attempts

    if len(valid_attempts) >= MAX_FAILED_ATTEMPTS:
        oldest_attempt = min(valid_attempts)
        remaining = int(RATE_LIMIT_WINDOW_SECONDS - (now - oldest_attempt))
        return False, max(1, remaining)

    return True, 0


def record_failed_login(ip: str, username: str) -> None:
    """Records a failed login timestamp for the given IP and username."""
    now = time.time()
    key = (ip, (username or "").strip().lower())
    if key not in _failed_login_attempts:
        _failed_login_attempts[key] = []
    _failed_login_attempts[key].append(now)


def clear_failed_logins(ip: str, username: str) -> None:
    """Clears failed login counter upon successful authentication."""
    key = (ip, (username or "").strip().lower())
    _failed_login_attempts.pop(key, None)


# ==============================================================================
# CSRF PROTECTION
# ==============================================================================

def generate_csrf_token() -> str:
    """Returns or generates a session-bound cryptographically random CSRF token."""
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


def validate_csrf_token(token: Optional[str]) -> bool:
    """Validates submitted CSRF token using constant-time comparison."""
    if not token or not isinstance(token, str):
        return False
    expected = session.get("_csrf_token")
    if not expected or not isinstance(expected, str):
        return False
    return hmac.compare_digest(token, expected)


# ==============================================================================
# RBAC & ACCESS CONTROL DECORATORS
# ==============================================================================

def login_required(f):
    """
    Requires an active authenticated session.
    Verifies that the account still exists and is_active == 1 in the database.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            if request.path.startswith("/api/") or request.is_json:
                return jsonify({"success": False, "error": "Authentication required", "status": 401}), 401
            return redirect(url_for("login", next=request.path))

        # Check account active status directly against the database
        db_path = current_app.config.get("DB_PATH")
        user = get_user_by_id(user_id, db_path=db_path)
        if not user or user.get("is_active") != 1:
            log_audit_event(
                action="DEACTIVATED_ACCOUNT_ACCESS_DENIED",
                status="DENIED",
                username=session.get("username"),
                user_id=user_id,
                ip_address=get_client_ip(),
                details=f"Access denied to deactivated/deleted account on {request.path}",
                db_path=db_path
            )
            session.clear()
            if request.path.startswith("/api/") or request.is_json:
                return jsonify({"success": False, "error": "Account deactivated", "status": 403}), 403
            return render_template("403.html", message="Your account has been deactivated. Please contact an administrator."), 403

        # Enforce mandatory password change server-side
        if user.get("must_change_password") == 1 and request.path not in ("/change-password", "/logout"):
            if request.path.startswith("/api/") or request.is_json:
                return jsonify({
                    "success": False,
                    "error": "Password change required before accessing system",
                    "status": 403,
                    "must_change_password": True
                }), 403
            return redirect(url_for("change_password"))

        # Update activity timestamp
        session["last_active"] = time.time()
        return f(*args, **kwargs)
    return decorated_function


def roles_required(*allowed_roles: str):
    """
    Requires the authenticated user to hold one of the specified roles.
    Enforces least privilege on the server side.
    """
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            user_role = session.get("role")
            if user_role not in allowed_roles:
                db_path = current_app.config.get("DB_PATH")
                log_audit_event(
                    action="ROLE_ACCESS_DENIED",
                    status="DENIED",
                    username=session.get("username"),
                    user_id=session.get("user_id"),
                    ip_address=get_client_ip(),
                    details=f"Role '{user_role}' denied access to {request.path} (Allowed: {allowed_roles})",
                    db_path=db_path
                )
                if request.path.startswith("/api/") or request.is_json:
                    return jsonify({
                        "success": False,
                        "error": "Forbidden: You do not have permission to access this resource",
                        "status": 403
                    }), 403
                return render_template(
                    "403.html",
                    message="Access Forbidden: Your account role does not have permission to access this section."
                ), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ==============================================================================
# MIDDLEWARE INITIALIZATION
# ==============================================================================

def init_auth_middleware(app):
    """Registers Jinja context processors and global request handlers."""

    @app.context_processor
    def inject_auth_context():
        """Exposes CSRF token and current user profile to all Jinja templates."""
        return {
            "csrf_token": generate_csrf_token,
            "current_user": {
                "is_authenticated": "user_id" in session,
                "user_id": session.get("user_id"),
                "username": session.get("username"),
                "full_name": session.get("full_name"),
                "role": session.get("role")
            }
        }

    @app.before_request
    def check_session_inactivity():
        """Enforces 30-minute idle session timeout."""
        if "user_id" in session:
            last_active = session.get("last_active")
            now = time.time()
            if last_active and (now - last_active) > IDLE_TIMEOUT_SECONDS:
                db_path = app.config.get("DB_PATH")
                log_audit_event(
                    action="SESSION_IDLE_TIMEOUT",
                    status="SUCCESS",
                    username=session.get("username"),
                    user_id=session.get("user_id"),
                    ip_address=get_client_ip(),
                    details="Session expired due to 30 minutes of inactivity",
                    db_path=db_path
                )
                session.clear()
                if request.path.startswith("/api/") or request.is_json:
                    return jsonify({"success": False, "error": "Session expired due to inactivity", "status": 401}), 401
                return redirect(url_for("login", message="Session expired due to inactivity. Please log in again."))

    @app.before_request
    def enforce_csrf_on_post():
        """
        Validates CSRF token on all POST requests.
        Allows exemption for public webhook/PWA if configured.
        """
        if not app.config.get("WTF_CSRF_ENABLED", True):
            return

        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            # Exclude endpoints where external clients may post (none currently public)
            # Accept token from form or header
            token = request.form.get("csrf_token") or request.headers.get("X-CSRFToken") or request.headers.get("X-CSRF-Token")
            
            # If payload is JSON and token is inside JSON body
            if not token and request.is_json:
                data = request.get_json(silent=True) or {}
                token = data.get("csrf_token")

            # Check validity
            if not validate_csrf_token(token):
                db_path = app.config.get("DB_PATH")
                log_audit_event(
                    action="CSRF_VALIDATION_FAILURE",
                    status="DENIED",
                    username=session.get("username"),
                    user_id=session.get("user_id"),
                    ip_address=get_client_ip(),
                    details=f"Missing or invalid CSRF token on POST {request.path}",
                    db_path=db_path
                )
                if request.path.startswith("/api/") or request.is_json:
                    return jsonify({"success": False, "error": "Invalid or missing CSRF token", "status": 400}), 400
                return render_template("403.html", message="Invalid or missing CSRF security token. Please refresh and try again."), 400
