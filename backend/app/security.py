import base64
import hashlib
import hmac
import json
import logging
from typing import Any

from fastapi import HTTPException, Request

from app.config import JWT_ALGORITHM, JWT_EXPIRES_SECONDS, JWT_SECRET
from app.db import get_user_by_id


logger = logging.getLogger(__name__)


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("utf-8"))


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _verify_hs256(token: str, secret: str) -> dict[str, Any] | None:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".", 2)
    except ValueError:
        return None

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    try:
        provided_sig = _b64url_decode(signature_b64)
    except Exception:
        return None

    if not hmac.compare_digest(provided_sig, expected_sig):
        return None

    try:
        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None

    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_int = int(exp)
        except Exception:
            return None
        now = int(__import__("time").time())
        if now >= exp_int:
            return None

    return payload


def create_access_token(*, user_id: int) -> str:
    if JWT_ALGORITHM.upper() != "HS256":
        raise RuntimeError("Unsupported JWT algorithm.")
    now = int(__import__("time").time())
    payload: dict[str, Any] = {"user_id": int(user_id), "iat": now, "exp": now + int(JWT_EXPIRES_SECONDS)}
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(signature)}"


def decode_jwt_optional(token: str) -> dict[str, Any] | None:
    if JWT_ALGORITHM.upper() != "HS256":
        return None
    return _verify_hs256(token, JWT_SECRET)


def get_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def get_optional_current_user(request: Request) -> dict[str, Any] | None:
    token = get_bearer_token(request)
    if not token:
        return None

    payload = decode_jwt_optional(token)
    if not payload:
        logger.info("Invalid bearer token provided; continuing as guest.")
        return None

    user_id = payload.get("user_id") or payload.get("id") or payload.get("sub")
    try:
        user_id_int = int(user_id)
    except Exception:
        logger.info("Bearer token missing a usable user id; continuing as guest.")
        return None

    user = get_user_by_id(user_id_int)
    if not user:
        logger.info("Bearer token user not found; continuing as guest.")
        return None

    return user


def get_current_user(request: Request) -> dict[str, Any]:
    user = get_optional_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token.")
    return user
