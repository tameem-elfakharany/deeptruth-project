import logging
import random
import numpy as np
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.predictions import router as predictions_router
from app.api.routes import router
from app.config import HEATMAPS_DIR, MODEL_PATH, ONNX_MODELS_DIR, OUTPUTS_DIR, UPLOADS_DIR
from app.db import init_db
from app.services.model_loader import load_audio_model, load_lipnet_model, load_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def _set_deterministic():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


KNOWN_GOOD_MODEL_MD5 = "1f07bc973f7079374f91c8ca84a77f42"


def _verify_model_integrity():
    import hashlib
    from app.config import PYTORCH_IMAGE_MODEL_PATH
    path = PYTORCH_IMAGE_MODEL_PATH
    if not path.exists():
        logger.warning("MODEL INTEGRITY: image model file not found at %s", path)
        return
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 == KNOWN_GOOD_MODEL_MD5:
        logger.info("MODEL INTEGRITY: OK (MD5 matches known-good model)")
    else:
        logger.warning(
            "MODEL INTEGRITY WARNING: MD5 mismatch! current=%s expected=%s — "
            "model may have been replaced. Detection accuracy may be degraded.",
            md5, KNOWN_GOOD_MODEL_MD5,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    _set_deterministic()
    _verify_model_integrity()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    init_db()
    
    # Load Image Model (ONNX preferred, TF fallback)
    model = load_model()
    app.state.model = model
    model_type = model.get('type', 'unknown') if isinstance(model, dict) else 'tensorflow'
    logger.info("Image model ready: type=%s", model_type)

    # Load Video Model (ONNX preferred, TF fallback)
    lipnet_model = load_lipnet_model()
    app.state.lipnet_model = lipnet_model
    video_type = lipnet_model.get('type', 'unknown') if isinstance(lipnet_model, dict) else 'none'
    logger.info("Video model ready: type=%s", video_type)

    # Load Audio Model (PyTorch)
    audio_model = load_audio_model()
    app.state.audio_model = audio_model
    audio_type = audio_model.get('type', 'unknown') if isinstance(audio_model, dict) else 'none'
    logger.info("Audio model ready: type=%s", audio_type)

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="DeepTruth Deepfake Detection API",
    description="Production FastAPI backend for binary deepfake image detection",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Incoming request: %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("Response status: %d", response.status_code)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://0.0.0.0:3000",
        "http://0.0.0.0:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(router)
app.include_router(auth_router)
app.include_router(predictions_router)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
