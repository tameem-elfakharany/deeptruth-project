from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]

# Legacy TF model paths (kept for backward compat)
MODEL_PATH = BASE_DIR / "models/xception_deepfake_classifier.h5"
LIPNET_MODEL_PATH = BASE_DIR / "models/lipnet_deepfake_classifier.h5"

# PyTorch model paths (take priority over ONNX and TF)
PYTORCH_IMAGE_MODEL_PATH = BASE_DIR / "models/deeptruth_image_model_final.pth"
PYTORCH_AUDIO_MODEL_PATH = BASE_DIR / "models/deeptruth_audio_model_final.pth"

# ONNX model paths (used when available, take priority over TF models)
ONNX_MODELS_DIR = BASE_DIR / "models/onnx_models"
ONNX_IMAGE_MODEL_PATH = ONNX_MODELS_DIR / "deeptruth_image_sim.onnx"
ONNX_IMAGE_FULL_MODEL_PATH = ONNX_MODELS_DIR / "deeptruth_image_full.onnx"
ONNX_VIDEO_MODEL_PATH = ONNX_MODELS_DIR / "deeptruth_video.onnx"
ONNX_DEPLOYMENT_META_PATH = ONNX_MODELS_DIR / "deployment_meta.json"

# ImageNet normalization (used by ONNX/PyTorch models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INPUT_HEIGHT = 224
INPUT_WIDTH = 224
INPUT_CHANNELS = 3

THRESHOLD = 0.5
AUDIO_THRESHOLD = 0.35   # Lower threshold for audio — model tends to under-predict fake probability

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_VIDEO_UPLOAD_BYTES = 100 * 1024 * 1024

UPLOADS_DIR = BASE_DIR / "backend" / "uploads"
OUTPUTS_DIR = BASE_DIR / "backend" / "outputs"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"

DB_PATH = BASE_DIR / "backend" / "deeptruth.db"

JWT_SECRET = os.getenv("DEEPTRUTH_JWT_SECRET") or os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY") or "deeptruth-dev-secret"
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_SECONDS = 60 * 60 * 24 * 7

