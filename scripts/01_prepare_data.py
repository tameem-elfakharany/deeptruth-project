# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
DeepTruth - Phase 1: Local Data Preparation Script
====================================================
Run this on your local PC. It will:
  1. Install all required packages
  2. Download GenImage subset  (DALL-E, Stable Diffusion, Midjourney) ~8GB
  3. Download CelebDF-v2                                               ~2.2GB
  4. Download FaceForensics++ compressed via your download.py          ~10GB
  5. Download DFDC Preview via Kaggle                                  ~10GB
  6. Extract 32 frames from every video (DFD + new datasets)
  7. Run MTCNN face crop on ALL images and frames
  8. Build train/val/test splits
  9. Zip the final processed faces ready for Google Drive upload

Usage:
    python scripts/01_prepare_data.py

Requirements:
    - Your FaceForensics++ download.py saved at:
      C:/Users/hp/Desktop/DeepTruth/scripts/ff_download.py
    - Your kaggle.json at:
      C:/Users/hp/.kaggle/kaggle.json
    - ~60GB free disk space on your PC during processing
      (raw downloads + processed faces, raw deleted after processing)

Output:
    data/processed_faces/   — MTCNN-cropped 224x224 face images
    data/sequences/         — (32, 224, 224, 3) video sequence .npz files
    data/splits/            — train / val / test folders
    data/upload_package.zip — everything needed for Google Drive (~4-6GB)
"""

import os
import sys
import subprocess
import shutil
import random
import json
import time
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION — edit paths here if needed
# ─────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent          # DeepTruth/
DATA         = ROOT / 'data'
RAW          = DATA / 'raw'
PROCESSED    = DATA / 'processed_faces'
SEQUENCES    = DATA / 'sequences'
SPLITS       = DATA / 'splits'
SCRIPTS      = ROOT / 'scripts'

FF_SCRIPT    = SCRIPTS / 'ff_download.py'            # your FaceForensics++ download.py
KAGGLE_JSON  = Path.home() / '.kaggle' / 'kaggle.json'

FRAMES_PER_VIDEO = 8
MTCNN_BATCH_SIZE = 16
RANDOM_SEED      = 42

# How many videos to download per FF++ manipulation type
# Full dataset is ~1000 per type. 300 gives good coverage without 130GB download.
FF_NUM_VIDEOS = 300

random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def header(text):
    print('\n' + '=' * 60)
    print(f'  {text}')
    print('=' * 60)


def step(text):
    print(f'\n>> {text}')


def ok(text):
    print(f'  [OK] {text}')


def warn(text):
    print(f'  [WARN] {text}')


def count_files(path, exts=('.jpg', '.jpeg', '.png')):
    if not Path(path).exists():
        return 0
    return sum(1 for f in Path(path).rglob('*') if f.suffix.lower() in exts)


def count_videos(path):
    if not Path(path).exists():
        return 0
    return sum(1 for f in Path(path).rglob('*') if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'))


# ─────────────────────────────────────────────
# STEP 0 — Install packages
# ─────────────────────────────────────────────
def install_packages():
    header('STEP 0 — Installing required packages')

    # Use system Python (PyTorch conflicts with TF in the backend venv)
    PYTHON = r'C:/Users/hp/AppData/Local/Programs/Python/Python310/python.exe'

    packages = [
        'torch torchvision --index-url https://download.pytorch.org/whl/cpu',
        'facenet-pytorch',
        'opencv-python',
        'gdown',
        'huggingface_hub',
        'kaggle',
        'tqdm',
        'Pillow',
        'numpy',
    ]

    for pkg in packages:
        step(f'Installing {pkg.split()[0]}...')
        result = subprocess.run(
            [PYTHON, '-m', 'pip', 'install', '-q'] + pkg.split(),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            warn(f'Failed to install {pkg}: {result.stderr[:200]}')
        else:
            ok(f'{pkg.split()[0]} ready')


# ─────────────────────────────────────────────
# STEP 1 — Verify existing data
# ─────────────────────────────────────────────
def verify_existing_data():
    header('STEP 1 — Verifying existing data')

    real_count = count_files(RAW / 'real')
    fake_count = count_files(RAW / 'fake')
    dfd_real   = count_videos(RAW / 'video' / 'DFD_original_sequences')
    dfd_fake   = count_videos(RAW / 'video' / 'DFD_manipulated_sequences')

    print(f'  Real images (FFHQ):        {real_count:,}  (expected ~50,000)')
    print(f'  Fake images (StyleGAN2):   {fake_count:,}  (expected ~50,000)')
    print(f'  DFD real videos:           {dfd_real:,}')
    print(f'  DFD fake videos:           {dfd_fake:,}')

    if real_count < 1000 or fake_count < 1000:
        warn('Very few images found. Make sure your data/raw/ folder is intact.')

    return real_count, fake_count, dfd_real, dfd_fake


# ─────────────────────────────────────────────
# STEP 2 — Download AI-generated face datasets
# ─────────────────────────────────────────────
def download_genimage():
    header('STEP 2 — Downloading AI-generated face datasets')
    print('  Downloads multiple Kaggle datasets covering GAN + diffusion fakes')

    GENIMAGE_OUT = RAW / 'genimage'
    os.makedirs(GENIMAGE_OUT, exist_ok=True)

    existing = count_files(GENIMAGE_OUT)
    if existing > 1000:
        ok(f'AI-generated images already downloaded ({existing:,} images)')
        return existing

    # Kaggle datasets covering AI-generated faces
    # Using 'kaggle' CLI directly (not python -m kaggle)
    datasets = [
        ('manjilkarki/deepfake-and-real-images',        'deepfake-real'),
        ('xhlulu/140k-real-and-fake-faces',             '140k-faces'),
        ('ciplab/real-and-fake-face-detection',         'ciplab-faces'),
    ]

    for dataset_id, subfolder in datasets:
        out_dir = GENIMAGE_OUT / subfolder
        if count_files(out_dir) > 500:
            ok(f'{dataset_id} already downloaded ({count_files(out_dir):,} images)')
            continue

        step(f'Downloading {dataset_id}...')
        os.makedirs(out_dir, exist_ok=True)
        result = subprocess.run(
            ['kaggle', 'datasets', 'download',
             '-d', dataset_id, '-p', str(out_dir), '--unzip'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            warn(f'Failed: {result.stderr[:150]}')
        else:
            ok(f'{dataset_id}: {count_files(out_dir):,} images')

    total = count_files(GENIMAGE_OUT)
    print(f'\n  AI-generated images total: {total:,}')
    return total


# ─────────────────────────────────────────────
# STEP 2b — Download OpenRL/DeepFakeFace
# ─────────────────────────────────────────────
def download_deepfakeface():
    header('STEP 2b — Downloading OpenRL/DeepFakeFace (diffusion face fakes)')
    print('  Source: HuggingFace OpenRL/DeepFakeFace | arXiv:2309.02218')
    print('  Size: ~5.1 GB | 120K face images | SD v1.5 inpainting + standard')

    OUT_DIR  = RAW / 'deepfakeface'
    real_dir = OUT_DIR / 'real'
    fake_dir = OUT_DIR / 'fake'

    if count_files(real_dir) > 5000 and count_files(fake_dir) > 5000:
        ok(f'OpenRL/DeepFakeFace already downloaded '
           f'({count_files(real_dir):,} real, {count_files(fake_dir):,} fake)')
        return

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    step('Downloading via HuggingFace Hub...')
    try:
        from huggingface_hub import snapshot_download
        import zipfile

        cache_dir = OUT_DIR / 'hf_cache'
        os.makedirs(cache_dir, exist_ok=True)

        local_path = snapshot_download(
            repo_id='OpenRL/DeepFakeFace',
            repo_type='dataset',
            cache_dir=str(cache_dir),
        )
        ok(f'Snapshot downloaded to: {local_path}')

        # Extract all ZIP files — route to real/ or fake/ by filename
        zip_files = list(Path(local_path).rglob('*.zip'))
        if not zip_files:
            # Also check directly inside cache
            zip_files = list(cache_dir.rglob('*.zip'))

        step(f'Found {len(zip_files)} ZIP file(s) — extracting...')
        for zf in zip_files:
            zname = zf.name.lower()
            dest  = real_dir if 'real' in zname else fake_dir
            step(f'  {zf.name} → {dest.name}/')
            with zipfile.ZipFile(str(zf), 'r') as z:
                z.extractall(str(dest))

        # Flatten any nested subdirectories into the target dir
        for d in (real_dir, fake_dir):
            for sub in list(d.iterdir()):
                if sub.is_dir():
                    for img in sub.iterdir():
                        if img.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                            img.rename(d / img.name)
                    try:
                        sub.rmdir()
                    except Exception:
                        pass

        ok(f'OpenRL/DeepFakeFace: {count_files(real_dir):,} real, '
           f'{count_files(fake_dir):,} fake faces')

    except Exception as e:
        warn(f'snapshot_download failed: {e}')
        warn('Trying huggingface-cli fallback...')
        HF_CACHE = OUT_DIR / 'hf_cache'
        os.makedirs(HF_CACHE, exist_ok=True)
        result = subprocess.run(
            ['huggingface-cli', 'download', 'OpenRL/DeepFakeFace',
             '--repo-type', 'dataset', '--local-dir', str(HF_CACHE)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            warn(f'huggingface-cli failed: {result.stderr[:200]}')
            warn('Skipping OpenRL/DeepFakeFace — check your internet connection.')
        else:
            ok('CLI download complete — please re-run to extract ZIPs.')


# ─────────────────────────────────────────────
# STEP 2c — Download DiffusionFace (Zenodo)
# ─────────────────────────────────────────────
def download_diffusionface():
    header('STEP 2c — Downloading DiffusionFace (Zenodo | arXiv:2403.18471)')
    print('  Source: zenodo.org/records/10865300  (CC-BY-4.0)')
    print('  Size: ~50 GB | 11 diffusion models | face forgery dataset')

    OUT_DIR = RAW / 'diffusionface'

    if count_files(OUT_DIR) > 5000:
        ok(f'DiffusionFace already downloaded ({count_files(OUT_DIR):,} images)')
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    import urllib.request
    import json
    import zipfile

    # Zenodo has two records for this dataset — Part 1 is the main one
    RECORD_IDS = ['10865300', '10865065']

    for record_id in RECORD_IDS:
        step(f'Fetching file list for Zenodo record {record_id}...')
        try:
            api_url = f'https://zenodo.org/api/records/{record_id}'
            req = urllib.request.Request(api_url,
                      headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data   = json.loads(resp.read())
            files  = data.get('files', [])
            ok(f'Record {record_id}: {len(files)} file(s)')
        except Exception as e:
            warn(f'Could not fetch Zenodo record {record_id}: {e}')
            continue

        for finfo in files:
            fname = finfo.get('key', finfo.get('filename', ''))
            fsize = finfo.get('size', 0) / (1024 ** 3)
            # Support both old and new Zenodo API shapes
            furl  = (finfo.get('links', {}).get('self')
                     or finfo.get('links', {}).get('download')
                     or f'https://zenodo.org/records/{record_id}/files/{fname}')

            out_path = OUT_DIR / fname

            if out_path.exists():
                ok(f'  {fname} already downloaded')
                # Still extract if not yet extracted
            elif fsize > 0:
                step(f'  Downloading {fname} ({fsize:.1f} GB)...')
                try:
                    def _reporthook(count, block, total):
                        pct = count * block * 100 // max(total, 1)
                        if count % 500 == 0:
                            print(f'\r    {pct}%', end='', flush=True)
                    urllib.request.urlretrieve(furl, str(out_path), _reporthook)
                    print()  # newline after progress
                    ok(f'  {fname} downloaded')
                except Exception as e:
                    warn(f'  Failed to download {fname}: {e}')
                    continue

            # Extract ZIP/tar archives
            if fname.endswith('.zip') and out_path.exists():
                step(f'  Extracting {fname}...')
                try:
                    with zipfile.ZipFile(str(out_path), 'r') as z:
                        z.extractall(str(OUT_DIR))
                    out_path.unlink()
                    ok(f'  {fname} extracted')
                except Exception as e:
                    warn(f'  Extraction failed for {fname}: {e}')

            elif fname.endswith(('.tar.gz', '.tar')) and out_path.exists():
                step(f'  Extracting {fname}...')
                try:
                    import tarfile
                    with tarfile.open(str(out_path)) as t:
                        t.extractall(str(OUT_DIR))
                    out_path.unlink()
                    ok(f'  {fname} extracted')
                except Exception as e:
                    warn(f'  Extraction failed for {fname}: {e}')

    total = count_files(OUT_DIR)
    ok(f'DiffusionFace total: {total:,} images in {OUT_DIR}')


# ─────────────────────────────────────────────
# STEP 3 — Download CelebDF-v2
# ─────────────────────────────────────────────
def download_celebdf():
    header('STEP 3 — Downloading CelebDF-v2 (high-quality celebrity face swaps)')
    print('  Size: ~2.2GB')

    import gdown

    CELEBDF_OUT = RAW / 'videos' / 'celebdf'
    real_dir = CELEBDF_OUT / 'real'
    fake_dir = CELEBDF_OUT / 'fake'

    if count_videos(real_dir) > 50 and count_videos(fake_dir) > 50:
        ok(f'CelebDF-v2 already downloaded ({count_videos(real_dir)} real, {count_videos(fake_dir)} fake videos)')
        return

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    step('Downloading CelebDF-v2 via Kaggle...')
    # CelebDF-v2 is available on Kaggle as a pre-packaged dataset
    result = subprocess.run(
        ['kaggle', 'datasets', 'download',
         '-d', 'reubensuju/celeb-df-v2',
         '-p', str(CELEBDF_OUT), '--unzip'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        warn(f'CelebDF Kaggle download failed: {result.stderr[:150]}')
        # Try gdown as fallback
        step('Trying gdown fallback for CelebDF-v2...')
        try:
            gdown.download_folder(
                'https://drive.google.com/drive/folders/1iml8GVEK1LHLiD8wkASVBIYbFUNvHjSM',
                output=str(real_dir), quiet=False,
            )
        except Exception as e:
            warn(f'gdown also failed: {e}')
    else:
        ok(f'CelebDF-v2 downloaded')

    print(f'\n  CelebDF-v2: {count_videos(real_dir)} real, {count_videos(fake_dir)} fake videos')


# ─────────────────────────────────────────────
# STEP 4 — Download FaceForensics++
# ─────────────────────────────────────────────
def download_faceforensics():
    header('STEP 4 — Downloading FaceForensics++ (compressed c23)')
    print('  Covers: Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures')
    print(f'  Downloading {FF_NUM_VIDEOS} videos per manipulation type (~10GB total)')

    FF_OUT = RAW / 'videos' / 'faceforensics'

    if not FF_SCRIPT.exists():
        warn(f'FaceForensics++ download script not found at: {FF_SCRIPT}')
        warn('Save your ff_download.py script there and re-run this step.')
        warn('Skipping FaceForensics++ for now.')
        return

    # Check if already downloaded
    existing = count_videos(FF_OUT)
    if existing > 500:
        ok(f'FaceForensics++ already downloaded ({existing} videos)')
        return

    os.makedirs(FF_OUT, exist_ok=True)

    step(f'Downloading all FF++ manipulation types ({FF_NUM_VIDEOS} videos each)...')
    print('  You will need to press a key to confirm the Terms of Service.')
    print('  The download will then run automatically.\n')

    PYTHON = r'C:/Users/hp/AppData/Local/Programs/Python/Python310/python.exe'
    cmd = [
        PYTHON, str(FF_SCRIPT),
        str(FF_OUT),
        '-d', 'all',
        '-c', 'c23',
        '-t', 'videos',
        '-n', str(FF_NUM_VIDEOS),
        '--server', 'EU2',
        '--auto_confirm',
    ]

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        warn('FaceForensics++ download encountered an error.')
        warn('Try running the download script manually if needed.')
    else:
        ok(f'FaceForensics++ downloaded: {count_videos(FF_OUT)} videos')


# ─────────────────────────────────────────────
# STEP 5 — Download DFDC Preview
# ─────────────────────────────────────────────
def download_dfdc():
    header('STEP 5 — Downloading DFDC Preview (diverse real-world deepfakes)')
    print('  Size: ~10GB | Covers: unconstrained in-the-wild deepfakes')

    DFDC_OUT = RAW / 'videos' / 'dfdc'

    if count_videos(DFDC_OUT) > 200:
        ok(f'DFDC already downloaded ({count_videos(DFDC_OUT)} videos)')
        return

    if not KAGGLE_JSON.exists():
        warn(f'Kaggle API key not found at {KAGGLE_JSON}')
        warn('Go to kaggle.com → Account → Create New Token → save kaggle.json')
        warn(f'Place it at: {KAGGLE_JSON}')
        warn('Skipping DFDC for now.')
        return

    os.makedirs(DFDC_OUT, exist_ok=True)

    step('Downloading DFDC Preview from Kaggle...')
    result = subprocess.run(
        ['kaggle', 'datasets', 'download',
         '-d', 'bardofcodes/deepfake-detection-challenge-dfdcp',
         '-p', str(DFDC_OUT), '--unzip'],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        warn(f'DFDC download failed: {result.stderr[:150]}')
        step('Trying alternative DFDC dataset...')
        result2 = subprocess.run(
            ['kaggle', 'datasets', 'download',
             '-d', 'sorokin8819/dfdc-deepfake-detection-challenge',
             '-p', str(DFDC_OUT), '--unzip'],
            capture_output=True, text=True
        )
        if result2.returncode != 0:
            warn(f'Alternative also failed: {result2.stderr[:150]}')
            warn('Skipping DFDC.')
            return

    ok(f'DFDC downloaded: {count_videos(DFDC_OUT)} videos')


# ─────────────────────────────────────────────
# STEP 6 — Extract frames from all videos
# ─────────────────────────────────────────────
def extract_all_frames():
    header('STEP 6 — Extracting frames from all video datasets')
    import cv2
    import numpy as np
    from tqdm import tqdm

    FRAMES_TMP = DATA / 'frames_tmp'

    video_sources = [
        # (video_dir,                                          frames_out_dir,              label,        is_fake)
        (RAW / 'video' / 'DFD_original_sequences',           FRAMES_TMP / 'dfd_real',     'DFD Real',    False),
        (RAW / 'video' / 'DFD_manipulated_sequences',        FRAMES_TMP / 'dfd_fake',     'DFD Fake',    True),
        (RAW / 'videos' / 'celebdf' / 'real',                FRAMES_TMP / 'celebdf_real', 'CelebDF Real',False),
        (RAW / 'videos' / 'celebdf' / 'fake',                FRAMES_TMP / 'celebdf_fake', 'CelebDF Fake',True),
        (RAW / 'videos' / 'faceforensics' / 'original_sequences',    FRAMES_TMP / 'ff_real', 'FF++ Real', False),
        (RAW / 'videos' / 'faceforensics' / 'manipulated_sequences', FRAMES_TMP / 'ff_fake', 'FF++ Fake', True),
        (RAW / 'videos' / 'dfdc',                            FRAMES_TMP / 'dfdc_fake',    'DFDC',        True),
    ]

    def extract_frames_from_video(video_path, output_dir, n=FRAMES_PER_VIDEO):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return 0
        indices = np.linspace(0, total - 1, min(n, total), dtype=int)
        saved = 0
        stem = video_path.stem
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                out = output_dir / f'{stem}_f{idx:06d}.jpg'
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
        cap.release()
        return saved

    total_frames = 0
    for video_dir, frames_out, label, is_fake in video_sources:
        if not Path(video_dir).exists():
            warn(f'Skipping {label} — not found: {video_dir}')
            continue

        videos = list(Path(video_dir).rglob('*.mp4')) + \
                 list(Path(video_dir).rglob('*.avi')) + \
                 list(Path(video_dir).rglob('*.mov'))

        if not videos:
            warn(f'No videos found in {video_dir}')
            continue

        existing = count_files(frames_out)
        if existing > len(videos) * 5:
            ok(f'{label}: already extracted ({existing:,} frames)')
            total_frames += existing
            continue

        os.makedirs(frames_out, exist_ok=True)
        n_frames = 0
        for video in tqdm(videos, desc=label):
            n_frames += extract_frames_from_video(video, frames_out)

        ok(f'{label}: {len(videos)} videos → {n_frames:,} frames')
        total_frames += n_frames

    print(f'\n  Total extracted frames: {total_frames:,}')
    return FRAMES_TMP


# ─────────────────────────────────────────────
# STEP 7 — Face crop using OpenCV Haar Cascade
# ─────────────────────────────────────────────
def run_mtcnn_face_crop(frames_tmp_dir):
    header('STEP 7 — Face detection and cropping (224×224) via OpenCV')
    print('  Uses OpenCV built-in Haar cascade — no extra dependencies.')
    print('  Progress is saved — safe to interrupt and resume.\n')

    import cv2
    import numpy as np
    from tqdm import tqdm

    # Built-in cascade — ships with every OpenCV install
    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    MARGIN = 0.25   # 25% padding around detected face box
    OUT_SIZE = 224

    def crop_face(img_bgr):
        """
        Detect the largest face in img_bgr and return a 224×224 crop.
        Falls back to a centre crop if no face is detected.
        Returns None if image is unreadable.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
        )

        h, w = img_bgr.shape[:2]

        if len(faces) == 0:
            # No face detected — fall back to centre square crop
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            crop = img_bgr[y0:y0+side, x0:x0+side]
        else:
            # Pick the largest face
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, fw, fh = faces[0]
            # Add margin
            pad_x = int(fw * MARGIN)
            pad_y = int(fh * MARGIN)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w, x + fw + pad_x)
            y1 = min(h, y + fh + pad_y)
            crop = img_bgr[y0:y1, x0:x1]

        if crop.size == 0:
            return None
        return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

    def process_directory(input_dir, output_dir, label):
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            return 0, 0

        files = [f for f in input_dir.rglob('*')
                 if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

        if not files:
            return 0, 0

        os.makedirs(output_dir, exist_ok=True)

        # Skip already-processed files (resumable)
        existing = {f.name for f in output_dir.iterdir()}
        files = [f for f in files if f.name not in existing]

        if not files:
            n = count_files(output_dir)
            ok(f'{label}: already processed ({n:,} files)')
            return n, 0

        saved = skipped = 0
        for fpath in tqdm(files, desc=label):
            img = cv2.imread(str(fpath))
            if img is None:
                skipped += 1
                continue
            crop = crop_face(img)
            if crop is None:
                skipped += 1
                continue
            out_path = output_dir / fpath.name
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        return saved, skipped

    # ── Static image datasets ──────────────────────────────────────
    static_tasks = [
        (RAW / 'real',                                PROCESSED / 'real' / 'ffhq',           'FFHQ real'),
        (RAW / 'fake',                                PROCESSED / 'fake' / 'stylegan2',       'StyleGAN2 fake'),
        (RAW / 'genimage' / 'stable_diffusion_v_1_4', PROCESSED / 'fake' / 'sd14',           'SD1.4'),
        (RAW / 'genimage' / 'dalle3',                 PROCESSED / 'fake' / 'dalle',          'DALL-E 3'),
        (RAW / 'genimage' / 'midjourney',             PROCESSED / 'fake' / 'midjourney',     'Midjourney'),
        # OpenRL/DeepFakeFace — diffusion-generated face fakes
        (RAW / 'deepfakeface' / 'real',               PROCESSED / 'real' / 'deepfakeface',   'DeepFakeFace real'),
        (RAW / 'deepfakeface' / 'fake',               PROCESSED / 'fake' / 'deepfakeface_sd','DeepFakeFace SD fake'),
        # DiffusionFace — 11 diffusion models, face forgery
        (RAW / 'diffusionface',                       PROCESSED / 'fake' / 'diffusionface',  'DiffusionFace'),
    ]

    # ── Frame directories from videos ─────────────────────────────
    FRAMES_TMP = Path(frames_tmp_dir)
    frame_tasks = [
        (FRAMES_TMP / 'dfd_real',     PROCESSED / 'real' / 'dfd',      'DFD real frames'),
        (FRAMES_TMP / 'dfd_fake',     PROCESSED / 'fake' / 'dfd',      'DFD fake frames'),
        (FRAMES_TMP / 'celebdf_real', PROCESSED / 'real' / 'celebdf',  'CelebDF real frames'),
        (FRAMES_TMP / 'celebdf_fake', PROCESSED / 'fake' / 'celebdf',  'CelebDF fake frames'),
        (FRAMES_TMP / 'ff_real',      PROCESSED / 'real' / 'ff',       'FF++ real frames'),
        (FRAMES_TMP / 'ff_fake',      PROCESSED / 'fake' / 'ff',       'FF++ fake frames'),
        (FRAMES_TMP / 'dfdc_fake',    PROCESSED / 'fake' / 'dfdc',     'DFDC fake frames'),
    ]

    grand_saved = grand_skipped = 0
    for inp, out, label in static_tasks + frame_tasks:
        s, sk = process_directory(inp, out, label)
        grand_saved   += s
        grand_skipped += sk
        if s > 0:
            print(f'  {label}: {s:,} faces saved, {sk:,} skipped')

    print(f'\n  Face crop complete.')
    print(f'  Total face crops: {grand_saved:,}')
    print(f'  Total fallback/skipped: {grand_skipped:,}')


# ─────────────────────────────────────────────
# STEP 8 — Build video sequences (.npz)
# ─────────────────────────────────────────────
def build_video_sequences():
    header('STEP 8 — Building video sequences for temporal training')
    print('  Each .npz = 32 aligned face frames from one video')
    print('  Shape: (32, 224, 224, 3) — used by the temporal stream\n')

    import cv2
    import numpy as np
    from tqdm import tqdm

    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    MARGIN = 0.25

    def haar_crop(img_bgr, size=224):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40,40))
        h, w = img_bgr.shape[:2]
        if len(faces) == 0:
            side = min(h, w)
            y0, x0 = (h-side)//2, (w-side)//2
            crop = img_bgr[y0:y0+side, x0:x0+side]
        else:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, fw, fh = faces[0]
            px, py = int(fw*MARGIN), int(fh*MARGIN)
            x0,y0 = max(0,x-px), max(0,y-py)
            x1,y1 = min(w,x+fw+px), min(h,y+fh+py)
            crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    SEQ_REAL = SEQUENCES / 'real'
    SEQ_FAKE = SEQUENCES / 'fake'
    os.makedirs(SEQ_REAL, exist_ok=True)
    os.makedirs(SEQ_FAKE, exist_ok=True)

    video_seq_sources = [
        (RAW / 'video' / 'DFD_original_sequences',           SEQ_REAL, 'DFD Real',     500),
        (RAW / 'video' / 'DFD_manipulated_sequences',        SEQ_FAKE, 'DFD Fake',     500),
        (RAW / 'videos' / 'celebdf' / 'real',                SEQ_REAL, 'CelebDF Real', 300),
        (RAW / 'videos' / 'celebdf' / 'fake',                SEQ_FAKE, 'CelebDF Fake', 300),
        (RAW / 'videos' / 'faceforensics' / 'original_sequences',    SEQ_REAL, 'FF++ Real', 400),
        (RAW / 'videos' / 'faceforensics' / 'manipulated_sequences', SEQ_FAKE, 'FF++ Fake', 400),
        (RAW / 'videos' / 'dfdc',                            SEQ_FAKE, 'DFDC',         400),
    ]

    def video_to_sequence(video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < FRAMES_PER_VIDEO:
            cap.release()
            return None
        indices = np.linspace(0, total - 1, FRAMES_PER_VIDEO, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            crop = haar_crop(frame)
            if crop is None:
                cap.release()
                return None
            # Convert BGR→RGB
            frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames, axis=0)  # (32, 224, 224, 3)

    total_seqs = 0
    for video_dir, seq_out, label, max_v in video_seq_sources:
        if not Path(video_dir).exists():
            warn(f'Skipping {label} — not found')
            continue

        videos = (list(Path(video_dir).rglob('*.mp4')) +
                  list(Path(video_dir).rglob('*.avi')))[:max_v]

        saved = 0
        for vp in tqdm(videos, desc=f'Sequences: {label}'):
            seq_path = seq_out / f'{vp.stem}.npz'
            if seq_path.exists():
                saved += 1
                continue
            seq = video_to_sequence(vp)
            if seq is not None:
                np.savez_compressed(str(seq_path), frames=seq)
                saved += 1

        ok(f'{label}: {saved} sequences saved')
        total_seqs += saved

    print(f'\n  Total sequences: {total_seqs}')
    print(f'  Real: {len(list(SEQ_REAL.glob("*.npz")))}')
    print(f'  Fake: {len(list(SEQ_FAKE.glob("*.npz")))}')


# ─────────────────────────────────────────────
# STEP 9 — Build train / val / test splits
# ─────────────────────────────────────────────
def build_splits():
    header('STEP 9 — Building train / val / test splits')

    for split in ('train', 'val', 'test'):
        for label in ('real', 'fake'):
            os.makedirs(SPLITS / split / label, exist_ok=True)

    def collect(base):
        return [f for f in Path(base).rglob('*')
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    all_real = collect(PROCESSED / 'real')
    all_fake = collect(PROCESSED / 'fake')

    print(f'  Total real face images: {len(all_real):,}')
    print(f'  Total fake face images: {len(all_fake):,}')

    # Balance
    n = min(len(all_real), len(all_fake))
    all_real = random.sample(all_real, n)
    all_fake = random.sample(all_fake, n)
    print(f'  Balanced to: {n:,} per class')

    def split_list(lst):
        random.shuffle(lst)
        t = int(n * 0.80)
        v = int(n * 0.10)
        return lst[:t], lst[t:t+v], lst[t+v:]

    real_train, real_val, real_test = split_list(all_real)
    fake_train, fake_val, fake_test = split_list(all_fake)

    def copy_files(files, dest, desc):
        from tqdm import tqdm
        os.makedirs(dest, exist_ok=True)
        for src in tqdm(files, desc=desc, leave=False):
            tag  = src.parent.name
            dst  = Path(dest) / f'{tag}_{src.name}'
            if not dst.exists():
                shutil.copy2(src, dst)

    copy_files(real_train, SPLITS / 'train' / 'real', 'real → train')
    copy_files(real_val,   SPLITS / 'val'   / 'real', 'real → val')
    copy_files(real_test,  SPLITS / 'test'  / 'real', 'real → test')
    copy_files(fake_train, SPLITS / 'train' / 'fake', 'fake → train')
    copy_files(fake_val,   SPLITS / 'val'   / 'fake', 'fake → val')
    copy_files(fake_test,  SPLITS / 'test'  / 'fake', 'fake → test')

    print('\n  === Split Summary ===')
    for split in ('train', 'val', 'test'):
        r = count_files(SPLITS / split / 'real')
        f = count_files(SPLITS / split / 'fake')
        print(f'  {split:5s}: {r:,} real + {f:,} fake = {r+f:,} total')


# ─────────────────────────────────────────────
# STEP 10 — Package for Drive upload
# ─────────────────────────────────────────────
def package_for_upload():
    header('STEP 10 — Packaging for Google Drive upload')

    UPLOAD_DIR  = DATA / 'drive_upload'
    UPLOAD_ZIP  = DATA / 'upload_package.zip'

    if UPLOAD_ZIP.exists():
        ok(f'Package already exists: {UPLOAD_ZIP}')
        size_gb = UPLOAD_ZIP.stat().st_size / (1024**3)
        print(f'  Size: {size_gb:.2f} GB')
        return

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    step('Copying splits...')
    if (SPLITS).exists():
        shutil.copytree(SPLITS, UPLOAD_DIR / 'splits', dirs_exist_ok=True)

    step('Copying sequences...')
    if SEQUENCES.exists():
        shutil.copytree(SEQUENCES, UPLOAD_DIR / 'sequences', dirs_exist_ok=True)

    step('Creating zip archive...')
    shutil.make_archive(str(DATA / 'upload_package'), 'zip', str(UPLOAD_DIR))

    size_gb = UPLOAD_ZIP.stat().st_size / (1024**3)
    ok(f'Package ready: {UPLOAD_ZIP}')
    print(f'  Size: {size_gb:.2f} GB')
    print(f'\n  Upload this zip to Google Drive, then extract it there.')
    print(f'  The Colab training notebooks will use it directly.')

    # Cleanup temp frames to free disk space
    step('Cleaning up temporary frame files...')
    tmp = DATA / 'frames_tmp'
    if tmp.exists():
        shutil.rmtree(tmp)
        ok('Temporary frames deleted')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print('=' * 60)
    print('  DeepTruth — Data Preparation Pipeline')
    print('  Estimated total time: 4-8 hours')
    print('  Safe to interrupt and resume at any step.')
    print('=' * 60)

    t_start = time.time()

    # Run all steps
    install_packages()
    verify_existing_data()
    download_genimage()
    download_deepfakeface()    # NEW — diffusion face fakes (HuggingFace, ~5 GB)
    download_diffusionface()   # NEW — 11 diffusion models, Zenodo, ~50 GB
    download_celebdf()
    download_faceforensics()
    download_dfdc()
    frames_tmp = extract_all_frames()
    run_mtcnn_face_crop(DATA / 'frames_tmp')
    build_video_sequences()
    build_splits()
    package_for_upload()

    elapsed = (time.time() - t_start) / 3600
    header(f'ALL STEPS COMPLETE — Total time: {elapsed:.1f} hours')
    print(f'\n  Next step: Upload data/upload_package.zip to Google Drive')
    print(f'  Then open notebooks/02_build_model.ipynb in Colab Pro')


if __name__ == '__main__':
    main()
