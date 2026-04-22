import hashlib
import hmac
import secrets
import sqlite3
from typing import Any

from app.config import DB_PATH


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_db_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                original_filename TEXT NOT NULL,
                prediction_label TEXT NOT NULL,
                raw_prediction REAL NOT NULL,
                fake_probability REAL NOT NULL,
                real_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                explanation TEXT,
                heatmap_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        cols = conn.execute("PRAGMA table_info('prediction_records')").fetchall()
        col_names = {c["name"] for c in cols}
        if "user_id" not in col_names:
            conn.execute("ALTER TABLE prediction_records ADD COLUMN user_id INTEGER;")
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def _hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password must not be empty.")
    iterations = 210_000
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations_str, salt_hex, hash_hex = stored.split("$", 3)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


def create_user(*, full_name: str | None, email: str, password: str) -> int:
    password_hash = _hash_password(password)
    conn = get_db_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO users (full_name, email, password)
            VALUES (?, ?, ?)
            """,
            (full_name, email.lower().strip(), password_hash),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict[str, Any] | None:
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT id, full_name, email, password, created_at
            FROM users
            WHERE email = ?
            """,
            (email.lower().strip(),),
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT id, full_name, email, password, created_at
            FROM users
            WHERE id = ?
            """,
            (int(user_id),),
        ).fetchone()
        user = _row_to_dict(row)
        if not user:
            return None
        return {k: v for k, v in user.items() if k != "password"}
    finally:
        conn.close()


def verify_user_login(*, email: str, password: str) -> dict[str, Any] | None:
    user = get_user_by_email(email)
    if not user:
        return None
    if not _verify_password(password, user["password"]):
        return None
    return {k: v for k, v in user.items() if k != "password"}


def save_prediction(
    *,
    user_id: int | None,
    original_filename: str,
    prediction_label: str,
    raw_prediction: float,
    fake_probability: float,
    real_probability: float,
    confidence: float,
    explanation: str | None,
    heatmap_path: str | None,
) -> int:
    conn = get_db_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO prediction_records (
                user_id,
                original_filename,
                prediction_label,
                raw_prediction,
                fake_probability,
                real_probability,
                confidence,
                explanation,
                heatmap_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                original_filename,
                prediction_label,
                raw_prediction,
                fake_probability,
                real_probability,
                confidence,
                explanation,
                heatmap_path,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_prediction_history(user_id: int) -> list[dict[str, Any]]:
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                id,
                user_id,
                original_filename,
                prediction_label,
                raw_prediction,
                fake_probability,
                real_probability,
                confidence,
                explanation,
                heatmap_path,
                created_at
            FROM prediction_records
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_prediction_history_for_user(user_id: int) -> list[dict[str, Any]]:
    return get_prediction_history(user_id)


def get_prediction_by_id(prediction_id: int) -> dict[str, Any] | None:
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT
                id,
                user_id,
                original_filename,
                prediction_label,
                raw_prediction,
                fake_probability,
                real_probability,
                confidence,
                explanation,
                heatmap_path,
                created_at
            FROM prediction_records
            WHERE id = ?
            """,
            (int(prediction_id),),
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_recent_predictions(limit: int = 10) -> list[dict[str, Any]]:
    limit = int(limit)
    if limit <= 0:
        return []
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                id,
                user_id,
                original_filename,
                prediction_label,
                raw_prediction,
                fake_probability,
                real_probability,
                confidence,
                explanation,
                heatmap_path,
                created_at
            FROM prediction_records
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
