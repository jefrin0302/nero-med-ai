"""
manage_users.py

Administrative CLI Utility for NeuroMed AI.
Provides:
1. Secure CLI-only bootstrap of initial Administrator account (manage_users.py init-admin).
2. User account provisioning (clinicians and laboratory staff).
3. Online, crash-consistent SQLite database backup.

Never stores or prints passwords in plaintext.
"""

import os
import sys
import argparse
import getpass
import re
from typing import Optional

from database import (
    init_db,
    create_user,
    get_user_by_username,
    list_all_users,
    log_audit_event,
    backup_database,
    open_db,
    DEFAULT_DB_PATH
)


def validate_password_strength(password: str) -> bool:
    """Enforces minimum 12 characters, uppercase, lowercase, number, and symbol."""
    if len(password) < 12:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    if not re.search(r"[^A-Za-z0-9]", password):
        return False
    return True


def init_admin_command(args: argparse.Namespace) -> int:
    """Bootstraps the first administrator account via CLI."""
    db_path = args.db_path or os.environ.get("WILSON_DB_PATH", DEFAULT_DB_PATH)
    init_db(db_path)

    # Check if an administrator already exists
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin';")
        count = cursor.fetchone()[0]
        if count > 0:
            print("[ERROR] An administrator account already exists. Aborting.", file=sys.stderr)
            return 1

    username = args.username or os.environ.get("BOOTSTRAP_ADMIN_USERNAME")
    full_name = args.name or os.environ.get("BOOTSTRAP_ADMIN_NAME")
    password = args.password or os.environ.get("BOOTSTRAP_ADMIN_PASSWORD")

    # Interactive prompt if not supplied
    if not username:
        try:
            username = input("Enter administrator username: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.", file=sys.stderr)
            return 1

    if not full_name:
        try:
            full_name = input("Enter administrator full name: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.", file=sys.stderr)
            return 1

    if not password:
        try:
            password = getpass.getpass("Enter administrator password (min 12 chars, mixed case, number, symbol): ")
            confirm = getpass.getpass("Confirm administrator password: ")
            if password != confirm:
                print("[ERROR] Passwords do not match.", file=sys.stderr)
                return 1
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.", file=sys.stderr)
            return 1

    if not validate_password_strength(password):
        print(
            "[ERROR] Password must be at least 12 characters and include uppercase, lowercase, "
            "digits, and special characters.",
            file=sys.stderr
        )
        return 1

    try:
        user = create_user(
            username=username,
            password=password,
            full_name=full_name,
            role="admin",
            db_path=db_path
        )
        log_audit_event(
            action="CLI_ADMIN_BOOTSTRAP",
            status="SUCCESS",
            username=username,
            user_id=user["user_id"],
            details="Root administrator bootstrapped via CLI utility",
            db_path=db_path
        )
        print(f"[SUCCESS] Root Administrator '{username}' created successfully (User ID: {user['user_id']}).")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to create administrator: {e}", file=sys.stderr)
        return 1


def create_user_command(args: argparse.Namespace) -> int:
    """Creates a clinician or lab_staff account."""
    db_path = args.db_path or os.environ.get("WILSON_DB_PATH", DEFAULT_DB_PATH)
    init_db(db_path)

    username = args.username
    full_name = args.name
    role = args.role.lower().strip()
    password = args.password

    if role not in ("clinician", "lab_staff", "admin"):
        print("[ERROR] Role must be 'clinician', 'lab_staff', or 'admin'.", file=sys.stderr)
        return 1

    if not password:
        password = getpass.getpass(f"Enter password for {username}: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("[ERROR] Passwords do not match.", file=sys.stderr)
            return 1

    try:
        user = create_user(
            username=username,
            password=password,
            full_name=full_name,
            role=role,
            db_path=db_path
        )
        log_audit_event(
            action="CLI_USER_CREATED",
            status="SUCCESS",
            username=username,
            user_id=user["user_id"],
            details=f"User created via CLI with role: {role}",
            db_path=db_path
        )
        print(f"[SUCCESS] User '{username}' ({role}) created successfully.")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to create user: {e}", file=sys.stderr)
        return 1


def list_users_command(args: argparse.Namespace) -> int:
    """Lists existing user accounts."""
    db_path = args.db_path or os.environ.get("WILSON_DB_PATH", DEFAULT_DB_PATH)
    init_db(db_path)
    users = list_all_users(db_path=db_path)
    if not users:
        print("No users found.")
        return 0

    print(f"{'ID':<4} {'Username':<18} {'Role':<12} {'Status':<10} {'Full Name'}")
    print("-" * 65)
    for u in users:
        status = "Active" if u["is_active"] == 1 else "Disabled"
        print(f"{u['user_id']:<4} {u['username']:<18} {u['role']:<12} {status:<10} {u['full_name']}")
    return 0


def backup_command(args: argparse.Namespace) -> int:
    """Creates crash-consistent SQLite online backup."""
    db_path = args.db_path or os.environ.get("WILSON_DB_PATH", DEFAULT_DB_PATH)
    try:
        dest = backup_database(db_path=db_path, target_backup_path=args.target)
        print(f"[SUCCESS] Crash-consistent backup created: {dest}")
        return 0
    except Exception as e:
        print(f"[ERROR] Backup failed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroMed AI User & Administration CLI")
    parser.add_argument("--db-path", help="Path to SQLite database file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init-admin
    p_init = subparsers.add_parser("init-admin", help="Bootstrap root Administrator account")
    p_init.add_argument("--username", help="Administrator username")
    p_init.add_argument("--name", help="Administrator full name")
    p_init.add_argument("--password", help="Administrator password")
    p_init.set_defaults(func=init_admin_command)

    # create-user
    p_create = subparsers.add_parser("create-user", help="Create clinician or lab staff account")
    p_create.add_argument("--username", required=True, help="Username")
    p_create.add_argument("--name", required=True, help="Full Name")
    p_create.add_argument("--role", required=True, choices=["clinician", "lab_staff", "admin"], help="User role")
    p_create.add_argument("--password", help="Password (prompted if omitted)")
    p_create.set_defaults(func=create_user_command)

    # list-users
    p_list = subparsers.add_parser("list-users", help="List registered accounts")
    p_list.set_defaults(func=list_users_command)

    # backup-db
    p_backup = subparsers.add_parser("backup-db", help="Create online crash-consistent backup")
    p_backup.add_argument("--target", help="Destination backup file path")
    p_backup.set_defaults(func=backup_command)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
