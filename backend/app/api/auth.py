import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Request

from app.db import create_user, get_user_by_email, verify_user_login
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.security import create_access_token, get_current_user


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
async def register(payload: RegisterRequest) -> UserResponse:
    existing = get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered.")
    try:
        user_id = create_user(full_name=payload.full_name, email=str(payload.email), password=payload.password)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Email already registered.")
    except Exception as e:
        logger.exception("Registration error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

    user = get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=500, detail="User creation failed.")

    return UserResponse(id=user_id, full_name=user.get("full_name"), email=user["email"], created_at=str(user.get("created_at")))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest) -> TokenResponse:
    logger.info("Login attempt for email: %s", payload.email)
    user = verify_user_login(email=str(payload.email), password=payload.password)
    if not user:
        logger.warning("Login failed for email: %s", payload.email)
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    
    logger.info("Login successful for email: %s, user_id: %s", payload.email, user["id"])
    token = create_access_token(user_id=int(user["id"]))
    return TokenResponse(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def me(request: Request) -> UserResponse:
    logger.info("Fetching current user info")
    user = get_current_user(request)
    logger.info("Current user: %s", user.get("email"))
    return UserResponse(
        id=int(user["id"]),
        full_name=user.get("full_name"),
        email=user["email"],
        created_at=str(user.get("created_at")) if user.get("created_at") is not None else None,
    )

