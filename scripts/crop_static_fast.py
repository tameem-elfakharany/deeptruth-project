# -*- coding: utf-8 -*-
"""
crop_static_fast.py — Fast parallel face crop using all CPU cores.

Two modes per dataset:
  - face_centred=True  : images are already face-only → just centre-crop + resize (fast)
  - face_centred=False : images have varied framing   → Haar detect + crop (slower)

Resumable: already-processed files are skipped automatically.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process
import functools

ROOT      = Path(__file__).parent.parent
RAW       = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed_faces'
OUT_SIZE  = 224
MARGIN    = 0.25
NUM_WORKERS = max(1, cpu_count() - 2)   # leave 2 cores for OS + pipeline

# ── Task list: (input_dir, output_dir, label, face_centred) ───────────
# face_centred=True  → simple centre crop, no detection needed
# face_centred=False → run Haar face detector
TASKS = [
    (RAW / 'real',                       PROCESSED / 'real' / 'ffhq',              'FFHQ real (50k)',          True),
    (RAW / 'fake',                       PROCESSED / 'fake' / 'stylegan2',          'StyleGAN2 fake (50k)',     True),
    (RAW / 'genimage' / 'deepfake-real', PROCESSED / 'fake' / 'genimage_gan',       'GenImage GAN (190k)',      False),
    (RAW / 'deepfakeface' / 'fake',      PROCESSED / 'fake' / 'deepfakeface_sd',    'DeepFakeFace (120k)',      True),
    (RAW / 'diffusionface' / 'ADM',      PROCESSED / 'fake' / 'diffusionface_adm',  'DiffusionFace ADM',       True),
    (RAW / 'diffusionface' / 'DDIM',     PROCESSED / 'fake' / 'diffusionface_ddim', 'DiffusionFace DDIM',      True),
    (RAW / 'diffusionface' / 'PNDM',     PROCESSED / 'fake' / 'diffusionface_pndm', 'DiffusionFace PNDM',      True),
    (RAW / 'diffusionface' / 'LDM',      PROCESSED / 'fake' / 'diffusionface_ldm',  'DiffusionFace LDM',       True),
    (RAW / 'diffusionface' / 'DiffSwap', PROCESSED / 'fake' / 'diffusionface_swap', 'DiffusionFace DiffSwap',  True),
]


# ── Per-worker state (loaded once per process) ────────────────────────
_face_cascade = None

def _worker_init():
    global _face_cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    _face_cascade = cv2.CascadeClassifier(cascade_path)


def _process_one(args):
    """
    Process a single image. Detects ALL faces and saves each as a separate crop.
    Returns number of faces saved (0 on failure).

    For face-centred datasets (face_centred=True): one centre crop, no detection.
    For multi-face datasets (face_centred=False): detect all faces, save each as
      stem_face0.jpg, stem_face1.jpg, etc. Falls back to centre crop if none found.
    """
    src_path, dst_path, face_centred = args
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return 0

        h, w = img.shape[:2]
        stem = Path(src_path).stem
        ext  = '.jpg'
        out_dir = Path(dst_path).parent

        if face_centred:
            # Fast path: already face-only image — single centre crop
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            crop = img[y0:y0+side, x0:x0+side]
            if crop.size == 0:
                return 0
            out = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return 1
        else:
            # Haar detect ALL faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            if len(faces) == 0:
                # Fallback: centre crop saved to original dst_path
                side = min(h, w)
                y0 = (h - side) // 2
                x0 = (w - side) // 2
                crop = img[y0:y0+side, x0:x0+side]
                if crop.size == 0:
                    return 0
                out = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return 1

            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            saved = 0
            for i, (fx, fy, fw, fh) in enumerate(faces):
                pad_x = int(fw * MARGIN)
                pad_y = int(fh * MARGIN)
                x0 = max(0, fx - pad_x);  y0 = max(0, fy - pad_y)
                x1 = min(w, fx + fw + pad_x); y1 = min(h, fy + fh + pad_y)
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                out = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
                # face0 keeps original name, face1+ get suffix
                face_path = dst_path if i == 0 else out_dir / f"{stem}_face{i}{ext}"
                cv2.imwrite(str(face_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            return saved
    except Exception:
        return 0


def process_dataset(input_dir, output_dir, label, face_centred):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f'  [SKIP] {label} — not found: {input_dir}', flush=True)
        return 0

    files = [f for f in input_dir.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not files:
        print(f'  [SKIP] {label} — no images', flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in output_dir.iterdir()}
    todo = [f for f in files if f.name not in existing]

    if not todo:
        n = len(list(output_dir.iterdir()))
        print(f'  [DONE] {label}: already complete ({n:,})', flush=True)
        return n

    mode = 'centre-crop' if face_centred else 'haar+crop'
    print(f'\n  {label}  [{mode}]', flush=True)
    print(f'  Total: {len(files):,}  |  Remaining: {len(todo):,}  |  Skipped: {len(existing):,}', flush=True)

    args = [(src, output_dir / src.name, face_centred) for src in todo]

    saved = 0
    chunk = 500
    for i in range(0, len(args), chunk):
        batch = args[i:i+chunk]
        results = pool.map(_process_one, batch)
        saved += sum(r for r in results if r)
        pct = min(100, (i + len(batch)) / len(args) * 100)
        print(f'\r    {pct:5.1f}%  {i+len(batch):,}/{len(args):,}  saved={saved:,}', end='', flush=True)

    print(f'\n  => {saved:,} saved', flush=True)
    return saved


if __name__ == '__main__':
    print('=' * 60, flush=True)
    print(f'  DeepTruth — Fast Static Face Crop  ({NUM_WORKERS} workers)', flush=True)
    print(f'  Output: {PROCESSED}', flush=True)
    print('=' * 60, flush=True)

    total = 0
    with Pool(processes=NUM_WORKERS, initializer=_worker_init) as pool:
        # make pool accessible to process_dataset
        import builtins
        builtins.pool = pool

        for inp, out, label, fc in TASKS:
            total += process_dataset(inp, out, label, fc)

    real_count = sum(1 for _ in (PROCESSED / 'real').rglob('*') if _.is_file()) if (PROCESSED / 'real').exists() else 0
    fake_count = sum(1 for _ in (PROCESSED / 'fake').rglob('*') if _.is_file()) if (PROCESSED / 'fake').exists() else 0

    print('\n' + '=' * 60, flush=True)
    print(f'  DONE — {total:,} new faces saved this run', flush=True)
    print(f'  Real faces total : {real_count:,}', flush=True)
    print(f'  Fake faces total : {fake_count:,}', flush=True)
    print('=' * 60, flush=True)
