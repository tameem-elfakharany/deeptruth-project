import logging

from fastapi import APIRouter, HTTPException, Request

from app.db import get_prediction_by_id, get_prediction_history_for_user, get_recent_predictions
from app.security import get_current_user


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/me")
async def my_predictions(request: Request) -> list[dict]:
    user = get_current_user(request)
    return get_prediction_history_for_user(int(user["id"]))


@router.get("/recent")
async def recent_predictions(limit: int = 10) -> list[dict]:
    return get_recent_predictions(limit=limit)


@router.get("/{prediction_id}")
async def prediction_details(request: Request, prediction_id: int) -> dict:
    user = get_current_user(request)
    rec = get_prediction_by_id(int(prediction_id))
    if not rec:
        raise HTTPException(status_code=404, detail="Not found.")
    if rec.get("user_id") is None or int(rec.get("user_id")) != int(user["id"]):
        raise HTTPException(status_code=404, detail="Not found.")
    return rec

