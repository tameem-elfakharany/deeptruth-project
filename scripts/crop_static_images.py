# -*- coding: utf-8 -*-
"""
crop_static_images.py
Run face detection + 224x224 crop on all STATIC image datasets.
Does NOT need video frame extraction to be done first.
Output goes to data/processed_faces/ — same location as the main pipeline.
Resumable: already-cropped files are skipped.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT      = Path(__file__).parent.parent
RAW       = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed_faces'
MARGIN    = 0.25
OUT_SIZE  = 224

# ── Load Haar cascade ──────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
print(f'Cascade loaded: {CASCADE_PATH}')

# ── Static tasks: (input_dir, output_dir, label) ──────────────────────
# Only includes datasets we actually have on disk.
STATIC_TASKS = [
    # Real faces
    (RAW / 'real',                       PROCESSED / 'real' / 'ffhq',              'FFHQ real (50k)'),
    # GAN fakes
    (RAW / 'fake',                       PROCESSED / 'fake' / 'stylegan2',          'StyleGAN2 fake (50k)'),
    (RAW / 'genimage' / 'deepfake-real', PROCESSED / 'fake' / 'genimage_gan',       'GenImage GAN fake (190k)'),
    # Diffusion fakes — DeepFakeFace (HuggingFace / arXiv:2309.02218)
    (RAW / 'deepfakeface' / 'fake',      PROCESSED / 'fake' / 'deepfakeface_sd',    'DeepFakeFace diffusion (120k)'),
    # DiffusionFace — 5 selected models (Zenodo / arXiv:2403.18471)
    (RAW / 'diffusionface' / 'ADM',      PROCESSED / 'fake' / 'diffusionface_adm',  'DiffusionFace ADM'),
    (RAW / 'diffusionface' / 'DDIM',     PROCESSED / 'fake' / 'diffusionface_ddim', 'DiffusionFace DDIM'),
    (RAW / 'diffusionface' / 'PNDM',     PROCESSED / 'fake' / 'diffusionface_pndm', 'DiffusionFace PNDM'),
    (RAW / 'diffusionface' / 'LDM',      PROCESSED / 'fake' / 'diffusionface_ldm',  'DiffusionFace LDM'),
    (RAW / 'diffusionface' / 'DiffSwap', PROCESSED / 'fake' / 'diffusionface_swap', 'DiffusionFace DiffSwap'),
]


def crop_face(img_bgr):
    """Detect largest face, return 224x224 crop. Falls back to centre crop."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    h, w = img_bgr.shape[:2]
    if len(faces) == 0:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = img_bgr[y0:y0+side, x0:x0+side]
    else:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, fw, fh = faces[0]
        pad_x, pad_y = int(fw * MARGIN), int(fh * MARGIN)
        x0 = max(0, x - pad_x);  y0 = max(0, y - pad_y)
        x1 = min(w, x + fw + pad_x); y1 = min(h, y + fh + pad_y)
        crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


def process_directory(input_dir, output_dir, label):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f'  [SKIP] {label} — source not found: {input_dir}')
        return 0, 0

    files = [f for f in input_dir.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not files:
        print(f'  [SKIP] {label} — no images in {input_dir}')
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in output_dir.iterdir()}
    todo = [f for f in files if f.name not in existing]

    if not todo:
        n = sum(1 for _ in output_dir.iterdir())
        print(f'  [DONE] {label}: already complete ({n:,} files)')
        return n, 0

    print(f'\n  {label}')
    print(f'  Source : {input_dir}')
    print(f'  Output : {output_dir}')
    print(f'  Total  : {len(files):,}  |  Remaining: {len(todo):,}  |  Already done: {len(existing):,}')

    saved = skipped = 0
    for fpath in tqdm(todo, desc=f'  {label[:40]}', unit='img'):
        img = cv2.imread(str(fpath))
        if img is None:
            skipped += 1
            continue
        crop = crop_face(img)
        if crop is None:
            skipped += 1
            continue
        cv2.imwrite(str(output_dir / fpath.name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    print(f'  => {saved:,} saved, {skipped:,} skipped')
    return saved, skipped


def main():
    print('=' * 60)
    print('  DeepTruth — Static Image Face Crop')
    print(f'  Output: {PROCESSED}')
    print('=' * 60)

    grand_saved = grand_skipped = 0
    for inp, out, label in STATIC_TASKS:
        s, sk = process_directory(inp, out, label)
        grand_saved   += s
        grand_skipped += sk

    print('\n' + '=' * 60)
    print(f'  DONE — {grand_saved:,} faces saved, {grand_skipped:,} skipped')
    real_count = sum(1 for _ in (PROCESSED / 'real').rglob('*') if _.is_file()) if (PROCESSED / 'real').exists() else 0
    fake_count = sum(1 for _ in (PROCESSED / 'fake').rglob('*') if _.is_file()) if (PROCESSED / 'fake').exists() else 0
    print(f'  Real faces : {real_count:,}')
    print(f'  Fake faces : {fake_count:,}')
    print('=' * 60)


if __name__ == '__main__':
    main()
