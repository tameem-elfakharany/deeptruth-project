"""
prepare_video_sequences.py
--------------------------
Organizes all available video datasets into the folder structure expected by
notebook 04_train_video_model.ipynb on Google Colab:

  video_sequences/
    real/
      <video_id>/
        frame_000.jpg  frame_001.jpg  ...
    fake/
      Deepfakes/        <-- FaceForensics++ subtypes
      Face2Face/
      FaceSwap/
      NeuralTextures/
      DeepFakeDetection/
      FaceShifter/
      deepfaketimit/
      dfd_manipulated/

Datasets handled:
  1. frames_tmp/dfd_real            -- pre-extracted DFD real frames
  2. frames_tmp/dfd_fake            -- pre-extracted DFD fake frames
  3. DeepfakeTIMIT.tar.gz           -- .avi fake videos
  4. FaceForensics++_C23.zip.part   -- 1000 real + 6000 fake mp4 videos (COMPLETE)

Output:
  data/video_sequences/   (ready to zip and upload to Google Drive)

Usage:
  python scripts/prepare_video_sequences.py [--min-frames 4] [--fps 5]
  python scripts/prepare_video_sequences.py --skip-dfd --skip-timit   # FF++ only
  python scripts/prepare_video_sequences.py --ff-only                 # FF++ only

Run this BEFORE zipping and uploading deeptruth_data.zip to Google Drive.
"""

import os
import re
import shutil
import tarfile
import zipfile
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
FRAMES_TMP    = BASE_DIR / 'data' / 'frames_tmp'
RAW_VIDEO     = BASE_DIR / 'data' / 'raw' / 'video'
OUT_DIR       = BASE_DIR / 'data' / 'video_sequences'
TIMIT_ARCHIVE = RAW_VIDEO / 'DeepfakeTIMIT' / 'DeepfakeTIMIT.tar.gz'
TIMIT_EXTRACT = RAW_VIDEO / 'DeepfakeTIMIT' / 'extracted'
FF_C23_ZIP    = RAW_VIDEO / 'FaceForensicsC23' / 'FaceForensics++_C23.zip.part'

DFD_REAL_DIR  = FRAMES_TMP / 'dfd_real'
DFD_FAKE_DIR  = FRAMES_TMP / 'dfd_fake'


# ── Helpers ───────────────────────────────────────────────────────────

def sample_frames_from_video(video_path: str, dst_dir: Path, fps_sample: int) -> int:
    """Extract evenly-sampled frames from a video file. Returns frames saved."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(video_fps / fps_sample))
    dst_dir.mkdir(parents=True, exist_ok=True)
    saved, idx = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            cv2.imwrite(str(dst_dir / f'frame_{saved:04d}.jpg'), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def group_flat_frames(src_dir: Path, dst_parent: Path, min_frames: int) -> int:
    """
    Groups flat frame files (named <video_id>_f<N>.jpg) into per-video subfolders.
    Already-processed sequences are skipped.
    """
    if not src_dir.exists():
        print(f'  [skip] {src_dir} not found')
        return 0

    pattern = re.compile(r'^(.+?)_f(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
    groups: dict = defaultdict(list)
    for f in src_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            groups[m.group(1)].append((int(m.group(2)), f))

    written = 0
    for video_id, frame_list in groups.items():
        if len(frame_list) < min_frames:
            continue
        seq_dir = dst_parent / video_id
        if seq_dir.exists() and any(seq_dir.iterdir()):
            written += 1
            continue
        seq_dir.mkdir(parents=True, exist_ok=True)
        frame_list.sort(key=lambda x: x[0])
        for i, (_, src_file) in enumerate(frame_list):
            shutil.copy2(src_file, seq_dir / f'frame_{i:04d}{src_file.suffix.lower()}')
        written += 1
    return written


def extract_timit_frames(min_frames: int, fps_sample: int) -> int:
    """Extract frames from DeepfakeTIMIT .avi files into fake/deepfaketimit/."""
    if not TIMIT_ARCHIVE.exists():
        print(f'  [skip] {TIMIT_ARCHIVE} not found')
        return 0
    if not HAS_CV2:
        print('  [skip] opencv not installed — pip install opencv-python-headless')
        return 0

    if not TIMIT_EXTRACT.exists():
        print(f'  Extracting {TIMIT_ARCHIVE.name} ...')
        with tarfile.open(TIMIT_ARCHIVE) as tar:
            tar.extractall(TIMIT_EXTRACT)

    dst_parent = OUT_DIR / 'fake' / 'deepfaketimit'
    written = 0
    for avi_path in sorted(TIMIT_EXTRACT.rglob('*.avi')):
        rel = avi_path.relative_to(TIMIT_EXTRACT)
        video_id = '__'.join(rel.with_suffix('').parts)
        seq_dir = dst_parent / video_id
        if seq_dir.exists() and any(seq_dir.iterdir()):
            written += 1
            continue
        saved = sample_frames_from_video(str(avi_path), seq_dir, fps_sample)
        if saved < min_frames:
            shutil.rmtree(seq_dir, ignore_errors=True)
        else:
            written += 1
    return written


def extract_ff_c23_frames(min_frames: int, fps_sample: int) -> tuple:
    """
    Extracts frames from FaceForensics++_C23.zip.part directly (no full extraction).
    Streams each mp4 out of the zip into a temp file, samples frames, deletes temp.
    Returns (real_count, fake_count).
    """
    if not FF_C23_ZIP.exists():
        print(f'  [skip] {FF_C23_ZIP} not found')
        return 0, 0
    if not HAS_CV2:
        print('  [skip] opencv not installed — pip install opencv-python-headless')
        return 0, 0

    real_written = 0
    fake_written = 0

    with zipfile.ZipFile(FF_C23_ZIP) as zf:
        mp4_entries = [n for n in zf.namelist() if n.endswith('.mp4')]
        total = len(mp4_entries)
        print(f'  Found {total} mp4 videos in FaceForensicsC23 zip')

        for i, entry in enumerate(mp4_entries, 1):
            parts = entry.split('/')
            # structure: FaceForensics++_C23/real/<id>.mp4
            #            FaceForensics++_C23/fake/<subtype>/<id>.mp4
            if len(parts) < 3:
                continue

            is_real = parts[1] == 'real'
            if is_real:
                video_id = Path(parts[2]).stem
                seq_dir = OUT_DIR / 'real' / video_id
            else:
                if len(parts) < 4:
                    continue
                subtype  = parts[2]   # Deepfakes, Face2Face, etc.
                video_id = Path(parts[3]).stem
                seq_dir  = OUT_DIR / 'fake' / subtype / video_id

            if seq_dir.exists() and any(seq_dir.iterdir()):
                if is_real:
                    real_written += 1
                else:
                    fake_written += 1
                continue

            # Stream mp4 to temp file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(zf.read(entry))
                tmp_path = tmp.name

            try:
                saved = sample_frames_from_video(tmp_path, seq_dir, fps_sample)
                if saved < min_frames:
                    shutil.rmtree(seq_dir, ignore_errors=True)
                elif is_real:
                    real_written += 1
                else:
                    fake_written += 1
            finally:
                os.unlink(tmp_path)

            if i % 200 == 0 or i == total:
                print(f'  Progress: {i}/{total} videos  '
                      f'(real={real_written} fake={fake_written})')

    return real_written, fake_written


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-frames', type=int, default=4)
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second to sample from videos (default: 5)')
    parser.add_argument('--skip-dfd',   action='store_true', help='Skip DFD frames_tmp step')
    parser.add_argument('--skip-timit', action='store_true', help='Skip DeepfakeTIMIT step')
    parser.add_argument('--skip-ff',    action='store_true', help='Skip FaceForensicsC23 step')
    parser.add_argument('--ff-only',    action='store_true', help='Only process FaceForensicsC23')
    args = parser.parse_args()

    if args.ff_only:
        args.skip_dfd = True
        args.skip_timit = True

    print(f'\nOutput directory: {OUT_DIR}')
    print(f'Min frames per sequence: {args.min_frames}')
    print(f'Video sampling: {args.fps} fps\n')

    # ── 1. DFD real frames ────────────────────────────────────────────
    if not args.skip_dfd:
        print('Processing DFD real frames ...')
        n = group_flat_frames(DFD_REAL_DIR, OUT_DIR / 'real', args.min_frames)
        print(f'  -> {n} real sequences\n')

        print('Processing DFD fake frames ...')
        n = group_flat_frames(DFD_FAKE_DIR, OUT_DIR / 'fake' / 'dfd_manipulated', args.min_frames)
        print(f'  -> {n} fake sequences\n')

    # ── 2. DeepfakeTIMIT ─────────────────────────────────────────────
    if not args.skip_timit:
        print('Processing DeepfakeTIMIT videos ...')
        n = extract_timit_frames(args.min_frames, args.fps)
        print(f'  -> {n} fake sequences\n')

    # ── 3. FaceForensics++ C23 ────────────────────────────────────────
    if not args.skip_ff:
        print('Processing FaceForensics++ C23 (1000 real + 6000 fake videos) ...')
        print('  This will take a while — streaming from zip, sampling frames ...')
        r, f = extract_ff_c23_frames(args.min_frames, args.fps)
        print(f'  -> {r} real + {f} fake sequences from FF++ C23\n')

    # ── Summary ───────────────────────────────────────────────────────
    real_seqs = sum(1 for p in (OUT_DIR / 'real').rglob('frame_0000*') if p.is_file()) \
                if (OUT_DIR / 'real').exists() else 0
    fake_seqs = sum(1 for p in (OUT_DIR / 'fake').rglob('frame_0000*') if p.is_file()) \
                if (OUT_DIR / 'fake').exists() else 0

    print('=' * 50)
    print(f'Real sequences : {real_seqs}')
    print(f'Fake sequences : {fake_seqs}')
    print(f'Total          : {real_seqs + fake_seqs}')

    if (OUT_DIR / 'fake').exists():
        print('\nFake breakdown:')
        for sub in sorted((OUT_DIR / 'fake').iterdir()):
            if sub.is_dir():
                count = sum(1 for p in sub.rglob('frame_0000*') if p.is_file())
                print(f'  {sub.name:<25} {count}')

    print('\nNext steps:')
    print('  1. Zip: python -c "import shutil; shutil.make_archive(\'deeptruth_data\', \'zip\', \'data\', \'video_sequences\')"')
    print('  2. Upload deeptruth_data.zip  ->  Google Drive/DeepTruth/')
    print('  3. Upload model_arch.py       ->  Google Drive/DeepTruth/')
    print('  4. Run notebook 04_train_video_model.ipynb on Colab')


if __name__ == '__main__':
    main()
