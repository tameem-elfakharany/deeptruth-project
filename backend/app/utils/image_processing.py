import numpy as np
import cv2
from fastapi import HTTPException

from app.config import INPUT_HEIGHT, INPUT_WIDTH


def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Corrupted or unreadable image file.")
    return img_bgr


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def preprocess_rgb_for_model(img_rgb: np.ndarray) -> np.ndarray:
    img_rgb = cv2.resize(img_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
    img_rgb = img_rgb.astype("float32") / 255.0
    img_rgb = np.expand_dims(img_rgb, axis=0)
    return img_rgb


def preprocess_image_bytes(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    img_bgr = decode_image_bytes_to_bgr(image_bytes)
    img_rgb = bgr_to_rgb(img_bgr)
    model_input = preprocess_rgb_for_model(img_rgb)
    return model_input, img_rgb

