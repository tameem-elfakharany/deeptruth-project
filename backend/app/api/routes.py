import logging
import shutil
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import UPLOADS_DIR, AUDIO_THRESHOLD
from app.db import save_prediction
from app.security import get_optional_current_user
from app.schemas.prediction import PredictionResponse
from app.services.explainability import generate_gradcam_heatmap
from app.services.face_detector import detect_all_faces
from app.services.inference import format_prediction_response, predict_all_faces, predict_audio, predict_image, predict_video
from app.services.video_processing import extract_video_frames
from app.utils.image_processing import preprocess_image_bytes
from app.utils.validators import (
    sanitize_filename,
    validate_upload_bytes,
    validate_upload_file_metadata,
    validate_video_bytes,
    validate_video_upload_metadata,
)


logger = logging.getLogger(__name__)

router = APIRouter()


def _get_model(request: Request) -> dict:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Image model is not loaded.")
    return model


def _get_video_model(request: Request) -> dict:
    model = getattr(request.app.state, "lipnet_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Video model is not loaded.")
    return model


def _get_audio_model(request: Request) -> dict:
    model = getattr(request.app.state, "audio_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Audio model is not loaded.")
    return model


@router.get("/health")
async def health(request: Request) -> dict:
    model_bundle = getattr(request.app.state, "model", None)
    video_bundle = getattr(request.app.state, "lipnet_model", None)
    audio_bundle = getattr(request.app.state, "audio_model", None)

    def _bundle_info(b) -> dict:
        if b is None:
            return {"loaded": False}
        if isinstance(b, dict):
            return {"loaded": True, "type": b.get("type", "unknown")}
        return {"loaded": True, "type": "tensorflow"}

    return {
        "status": "ok",
        "image_model": _bundle_info(model_bundle),
        "video_model": _bundle_info(video_bundle),
        "audio_model": _bundle_info(audio_bundle),
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    logger.info("Request received: /predict filename=%s", file.filename)
    filename = validate_upload_file_metadata(file)
    file_bytes = await file.read()
    validate_upload_bytes(file_bytes)

    logger.info("Upload bytes: %d", len(file_bytes))
    _, original_rgb = preprocess_image_bytes(file_bytes)
    logger.info("Decoded image shape: %s", original_rgb.shape)
    model_bundle = _get_model(request)

    # Detect all faces — if any face is fake the whole image is flagged
    face_crops, face_boxes = detect_all_faces(original_rgb)
    logger.info("Faces detected: %d, boxes: %s", len(face_crops), face_boxes)
    result = predict_all_faces(model_bundle, face_crops)

    raw_prediction = result['raw_prediction']
    logger.info("Prediction: filename=%s raw=%.6f faces=%d type=%s", filename, raw_prediction,
                result['faces_detected'], result.get('fake_type', 'N/A'))

    response = format_prediction_response(
        filename=filename,
        raw_prediction=raw_prediction,
        heatmap_path=None,
        fake_type=result.get('fake_type'),
        fake_type_confidence=result.get('fake_type_confidence'),
        type_probabilities=result.get('type_probabilities'),
        faces_detected=result['faces_detected'],
        flagged_face_index=result.get('flagged_face_index'),
    )

    user = get_optional_current_user(request)
    try:
        save_prediction(
            user_id=user["id"] if user else None,
            original_filename=filename,
            prediction_label=response.prediction_label,
            raw_prediction=response.raw_prediction,
            fake_probability=response.fake_probability,
            real_probability=response.real_probability,
            confidence=response.confidence,
            explanation=response.explanation,
            heatmap_path=response.heatmap_path,
        )
    except Exception:
        logger.exception("Failed to save prediction for filename=%s", filename)

    return response


@router.post("/predict-video", response_model=PredictionResponse)
async def predict_video_endpoint(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    logger.info("Request received: /predict-video filename=%s", file.filename)
    filename = validate_video_upload_metadata(file)
    temp_filename = f"{uuid.uuid4()}_{filename}"
    temp_path = UPLOADS_DIR / temp_filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        video_bundle = _get_video_model(request)
        model_type = video_bundle.get('type', 'tensorflow') if isinstance(video_bundle, dict) else 'tensorflow'

        if model_type == 'onnx':
            # ONNX video model: expects (1, T, 3, H, W)
            n_frames = video_bundle.get('n_frames', 8)
            frames = _extract_frames_for_onnx(str(temp_path), n_frames)
            result = predict_video(video_bundle, frames)
        else:
            # Legacy TF LipNet: expects (1, 20, 100, 100, 3)
            frames = extract_video_frames(str(temp_path))
            if frames is None:
                raise HTTPException(status_code=400, detail="Failed to extract frames from video.")
            model_input = np.expand_dims(frames, axis=0)
            result = predict_video(video_bundle, model_input)

        raw_prediction = result['raw_prediction']
        logger.info("Video prediction: filename=%s raw=%.6f", filename, raw_prediction)

        response = format_prediction_response(
            filename=filename,
            raw_prediction=raw_prediction,
            fake_type=result.get('fake_type'),
            fake_type_confidence=result.get('fake_type_confidence'),
        )

        user = get_optional_current_user(request)
        try:
            save_prediction(
                user_id=user["id"] if user else None,
                original_filename=filename,
                prediction_label=response.prediction_label,
                raw_prediction=response.raw_prediction,
                fake_probability=response.fake_probability,
                real_probability=response.real_probability,
                confidence=response.confidence,
                explanation=response.explanation,
                heatmap_path=None,
            )
        except Exception:
            logger.exception("Failed to save video prediction for filename=%s", filename)

        return response

    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.post("/predict-with-heatmap", response_model=PredictionResponse)
async def predict_with_heatmap_endpoint(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    logger.info("Request received: /predict-with-heatmap filename=%s", file.filename)
    filename = validate_upload_file_metadata(file)
    file_bytes = await file.read()
    validate_upload_bytes(file_bytes)

    model_input, original_rgb = preprocess_image_bytes(file_bytes)
    model_bundle = _get_model(request)

    # Detect all faces — if any face is fake the whole image is flagged
    face_crops, _ = detect_all_faces(original_rgb)
    result = predict_all_faces(model_bundle, face_crops)
    raw_prediction = result['raw_prediction']

    # Grad-CAM: run on the flagged face crop (or full image for TF)
    heatmap_path = None
    model_type = model_bundle.get('type', 'tensorflow') if isinstance(model_bundle, dict) else 'tensorflow'
    if model_type == 'tensorflow':
        tf_model = model_bundle['session'] if isinstance(model_bundle, dict) else model_bundle
        flagged_idx = result.get('flagged_face_index') or 0
        flagged_crop = face_crops[flagged_idx]
        filename_stem = Path(sanitize_filename(filename)).stem
        heatmap_path = generate_gradcam_heatmap(
            model=tf_model,
            model_input=model_input,
            original_rgb=flagged_crop,
            filename_stem=filename_stem,
        )

    logger.info("Prediction (with heatmap): filename=%s raw=%.6f faces=%d heatmap=%s",
                filename, raw_prediction, result['faces_detected'], heatmap_path)

    response = format_prediction_response(
        filename=filename,
        raw_prediction=raw_prediction,
        heatmap_path=heatmap_path,
        fake_type=result.get('fake_type'),
        fake_type_confidence=result.get('fake_type_confidence'),
        type_probabilities=result.get('type_probabilities'),
        faces_detected=result['faces_detected'],
        flagged_face_index=result.get('flagged_face_index'),
    )

    user = get_optional_current_user(request)
    try:
        save_prediction(
            user_id=user["id"] if user else None,
            original_filename=filename,
            prediction_label=response.prediction_label,
            raw_prediction=response.raw_prediction,
            fake_probability=response.fake_probability,
            real_probability=response.real_probability,
            confidence=response.confidence,
            explanation=response.explanation,
            heatmap_path=response.heatmap_path,
        )
    except Exception:
        logger.exception("Failed to save prediction for filename=%s", filename)

    return response


@router.post("/predict-audio", response_model=PredictionResponse)
async def predict_audio_endpoint(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """Audio deepfake detection endpoint. Accepts wav/flac/mp3/ogg/m4a files."""
    logger.info("Request received: /predict-audio filename=%s", file.filename)

    # Validate
    filename = file.filename or "audio"
    ext = Path(filename).suffix.lower()
    if ext not in {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac'}:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {ext}")

    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:   # 50 MB max for audio
        raise HTTPException(status_code=400, detail="Audio file too large (max 50 MB).")

    audio_bundle = _get_audio_model(request)

    try:
        result = predict_audio(audio_bundle, file_bytes)
    except Exception as exc:
        logger.exception("Audio inference failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Audio inference error: {exc}")

    raw_prediction = result['raw_prediction']
    logger.info("Audio prediction: filename=%s raw=%.6f type=%s",
                filename, raw_prediction, result.get('fake_type', 'N/A'))

    response = format_prediction_response(
        filename=filename,
        raw_prediction=raw_prediction,
        threshold=AUDIO_THRESHOLD,
        fake_type=result.get('fake_type'),
        fake_type_confidence=result.get('fake_type_confidence'),
        type_probabilities=result.get('type_probabilities'),
    )

    user = get_optional_current_user(request)
    try:
        save_prediction(
            user_id=user["id"] if user else None,
            original_filename=filename,
            prediction_label=response.prediction_label,
            raw_prediction=response.raw_prediction,
            fake_probability=response.fake_probability,
            real_probability=response.real_probability,
            confidence=response.confidence,
            explanation=response.explanation,
            heatmap_path=None,
        )
    except Exception:
        logger.exception("Failed to save audio prediction for filename=%s", filename)

    return response


def _extract_frames_for_onnx(video_path: str, n_frames: int = 32) -> np.ndarray:
    """Extract and preprocess frames for the ONNX video model.
    Returns (1, T, 3, 224, 224) float32 array with ImageNet normalization.
    """
    import cv2
    from app.config import IMAGENET_MEAN, IMAGENET_STD

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame.transpose(2, 0, 1) - mean) / std  # (3, 224, 224)
        frames.append(frame)

    cap.release()
    # (T, 3, H, W) -> (1, T, 3, H, W)
    return np.stack(frames, axis=0)[np.newaxis, ...]
