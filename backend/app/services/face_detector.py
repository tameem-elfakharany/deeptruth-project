"""
face_detector.py
Detects ALL faces in an image and returns individual 224x224 crops.
Falls back to a single centre crop if no faces are found.
Used at inference time so every face in a multi-person image is checked.
"""
import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade: cv2.CascadeClassifier | None = None

OUT_SIZE   = 224
MARGIN     = 0.30   # 30% padding around each detected face box
MIN_FACE   = 40     # minimum face side length in pixels


def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if _face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {_CASCADE_PATH}")
    return _face_cascade


def detect_all_faces(img_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Detect every face in img_rgb (H, W, 3) uint8.

    Returns:
        crops  — list of (224, 224, 3) uint8 face crops, one per detected face
        boxes  — list of (x, y, w, h) bounding boxes in original image coordinates

    If no face is found, returns a single centre crop of the whole image
    so the model always gets at least one input.
    """
    cascade = _get_cascade()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect with two scaleFactor passes for better recall
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(MIN_FACE, MIN_FACE))
    if len(faces) == 0:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(MIN_FACE, MIN_FACE))

    h, w = img_bgr.shape[:2]

    if len(faces) == 0:
        # Fallback: centre-square crop → whole image treated as one face
        logger.debug("No faces detected — falling back to centre crop")
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img_bgr[y0:y0+side, x0:x0+side]
        crop_rgb = cv2.cvtColor(cv2.resize(crop, (OUT_SIZE, OUT_SIZE)), cv2.COLOR_BGR2RGB)
        return [crop_rgb], []

    # Sort largest face first so face[0] is the dominant face
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    crops: List[np.ndarray] = []
    boxes: List[Tuple[int, int, int, int]] = []

    for (fx, fy, fw, fh) in faces_sorted:
        pad_x = int(fw * MARGIN)
        pad_y = int(fh * MARGIN)
        x0 = max(0, fx - pad_x)
        y0 = max(0, fy - pad_y)
        x1 = min(w, fx + fw + pad_x)
        y1 = min(h, fy + fh + pad_y)

        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(cv2.resize(crop, (OUT_SIZE, OUT_SIZE)), cv2.COLOR_BGR2RGB)
        crops.append(crop_rgb)
        boxes.append((int(fx), int(fy), int(fw), int(fh)))

    if not crops:
        # All crops were degenerate — fallback
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img_bgr[y0:y0+side, x0:x0+side]
        crop_rgb = cv2.cvtColor(cv2.resize(crop, (OUT_SIZE, OUT_SIZE)), cv2.COLOR_BGR2RGB)
        return [crop_rgb], []

    logger.debug("Detected %d face(s)", len(crops))
    return crops, boxes
