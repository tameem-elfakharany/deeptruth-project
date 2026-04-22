import re
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS, MAX_UPLOAD_BYTES, MAX_VIDEO_UPLOAD_BYTES


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "").name.strip()
    if not name:
        return "upload"
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name[:200] if len(name) > 200 else name


def validate_upload_file_metadata(file: UploadFile) -> str:
    filename = sanitize_filename(file.filename or "")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext or '(missing extension)'} not allowed. Allowed types: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )
    return filename


def validate_video_upload_metadata(file: UploadFile) -> str:
    filename = sanitize_filename(file.filename or "")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext or '(missing extension)'} not allowed. Allowed video types: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )
    return filename


def validate_upload_bytes(file_bytes: bytes) -> None:
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_UPLOAD_BYTES} bytes.",
        )


def validate_video_bytes(file_bytes: bytes) -> None:
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty video file upload.")
    if len(file_bytes) > MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Video file too large. Max size is {MAX_VIDEO_UPLOAD_BYTES} bytes.",
        )

