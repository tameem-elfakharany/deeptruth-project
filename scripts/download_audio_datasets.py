"""
DeepTruth — Audio Deepfake Dataset Downloader (Full Suite)
Downloads:
  1.  WaveFake         (Zenodo 5642694)       ~27 GB  -- neural vocoder fakes
  2.  ASVspoof 2021 LA (Zenodo 4837263)       ~7.7 GB -- TTS/voice conversion
  3.  ASVspoof 2019 LA (Edinburgh DataShare)  ~7.6 GB -- 19 TTS systems
  4.  ASVspoof 5       (Zenodo 14498691)      ~10 GB  -- adversarial + crowdsourced
  5.  CFAD             (Zenodo 8122764)       ~35 GB  -- Chinese deepfake audio
  6.  CVoiceFake       (Zenodo 11229569)      ~8 GB   -- voice cloning, 5 languages
  7.  DFADD            (HuggingFace)          ~5 GB   -- diffusion/flow-matching fakes
  8.  MLAAD            (HuggingFace)          ~15 GB  -- 51 languages, 140 TTS models

Run: python scripts/download_audio_datasets.py
"""

import os
import zipfile
import tarfile
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw" / "audio"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def curl_download(url, dest: Path, desc=''):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    if dest.exists():
        print(f'  [SKIP] {dest.name} ({dest.stat().st_size/1e9:.2f} GB)')
        return dest
    print(f'  Downloading {desc or dest.name} ...')
    cmd = ['curl', '-L', '-C', '-', '--retry', '5', '--retry-delay', '10',
           '--retry-max-time', '300', '-o', str(part), url]
    result = subprocess.run(cmd)
    if result.returncode == 0 and part.exists():
        part.rename(dest)
        print(f'  [OK] {dest.name} ({dest.stat().st_size/1e9:.2f} GB)')
    else:
        print(f'  [ERROR] curl failed for {url}')
    return dest


def extract(archive: Path, dest: Path, marker: str = None):
    check = dest / marker if marker else None
    if check and check.exists():
        print(f'  [SKIP] already extracted')
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f'  Extracting {archive.name} -> {dest} ...')
    try:
        if archive.suffix == '.zip':
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        elif archive.name.endswith(('.tar.gz', '.tar.bz2', '.tgz')):
            with tarfile.open(archive) as t:
                t.extractall(dest)
        print(f'  [OK] Extracted')
    except Exception as e:
        print(f'  [ERROR] {e}')


def hf_download(repo_id, dest: Path, repo_type='dataset'):
    """Download from HuggingFace using huggingface_hub."""
    if dest.exists() and any(dest.iterdir()):
        print(f'  [SKIP] {dest.name} already downloaded')
        return
    print(f'  Downloading {repo_id} from HuggingFace ...')
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        print(f'  [OK] {repo_id} downloaded to {dest}')
    except Exception as e:
        print(f'  [ERROR] HuggingFace download failed: {e}')
        print(f'  Install: pip install huggingface_hub')
        print(f'  Manual:  huggingface-cli download {repo_id} --repo-type {repo_type} --local-dir {dest}')


# ═══════════════════════════════════════════════════════
# 1. WaveFake (~27 GB) — Zenodo 5642694
# ═══════════════════════════════════════════════════════
def download_wavefake():
    print('\n' + '='*60)
    print('1. WaveFake (~27 GB) — neural vocoder fakes')
    print('='*60)
    dest_dir = DATA_DIR / 'WaveFake'
    dest_dir.mkdir(exist_ok=True)
    archive = dest_dir / 'generated_audio.zip'
    curl_download(
        'https://zenodo.org/records/5642694/files/generated_audio.zip?download=1',
        archive, 'WaveFake generated_audio.zip'
    )
    if archive.exists() and archive.stat().st_size > 1e6:
        extract(archive, dest_dir, 'generated_audio')


# ═══════════════════════════════════════════════════════
# 2. ASVspoof 2021 LA eval (~7.7 GB) — Zenodo 4837263
# ═══════════════════════════════════════════════════════
def download_asvspoof2021():
    print('\n' + '='*60)
    print('2. ASVspoof 2021 LA eval (~7.7 GB)')
    print('='*60)
    dest_dir = DATA_DIR / 'ASVspoof2021'
    dest_dir.mkdir(exist_ok=True)
    archive = dest_dir / 'ASVspoof2021_LA_eval.tar.gz'
    curl_download(
        'https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1',
        archive, 'ASVspoof2021_LA_eval.tar.gz'
    )
    if archive.exists() and archive.stat().st_size > 1e6:
        extract(archive, dest_dir, 'ASVspoof2021_LA_eval')


# ═══════════════════════════════════════════════════════
# 3. ASVspoof 2019 LA (~7.6 GB) — Edinburgh DataShare
# ═══════════════════════════════════════════════════════
def download_asvspoof2019():
    print('\n' + '='*60)
    print('3. ASVspoof 2019 LA (~7.6 GB) — 19 TTS systems')
    print('='*60)
    dest_dir = DATA_DIR / 'ASVspoof2019'
    dest_dir.mkdir(exist_ok=True)
    archive = dest_dir / 'LA.zip'
    curl_download(
        'https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip',
        archive, 'ASVspoof2019 LA.zip'
    )
    if archive.exists() and archive.stat().st_size > 1e6:
        extract(archive, dest_dir, 'LA')


# ═══════════════════════════════════════════════════════
# 4. ASVspoof 5 (~10 GB) — Zenodo 14498691
#    Latest edition: adversarial attacks + crowdsourced
# ═══════════════════════════════════════════════════════
def download_asvspoof5():
    print('\n' + '='*60)
    print('4. ASVspoof 5 (~10 GB) — adversarial attacks (2025)')
    print('='*60)
    dest_dir = DATA_DIR / 'ASVspoof5'
    dest_dir.mkdir(exist_ok=True)

    files = [
        ('https://zenodo.org/records/14498691/files/ASVspoof5.dev.tar.gz?download=1',
         'ASVspoof5.dev.tar.gz'),
        ('https://zenodo.org/records/14498691/files/ASVspoof5.eval.tar.gz?download=1',
         'ASVspoof5.eval.tar.gz'),
    ]
    for url, fname in files:
        archive = dest_dir / fname
        curl_download(url, archive, fname)
        if archive.exists() and archive.stat().st_size > 1e6:
            extract(archive, dest_dir)


# ═══════════════════════════════════════════════════════
# 5. CFAD (~35 GB) — Zenodo 8122764
#    Chinese Fake Audio Detection — 4 part zip
# ═══════════════════════════════════════════════════════
def download_cfad():
    print('\n' + '='*60)
    print('5. CFAD (~35 GB) — Chinese deepfake audio')
    print('='*60)
    dest_dir = DATA_DIR / 'CFAD'
    dest_dir.mkdir(exist_ok=True)

    parts = [
        ('https://zenodo.org/records/8122764/files/CFAD.zip?download=1',  'CFAD.zip'),
        ('https://zenodo.org/records/8122764/files/CFAD.z01?download=1',  'CFAD.z01'),
        ('https://zenodo.org/records/8122764/files/CFAD.z02?download=1',  'CFAD.z02'),
        ('https://zenodo.org/records/8122764/files/CFAD.z03?download=1',  'CFAD.z03'),
    ]
    for url, fname in parts:
        curl_download(url, dest_dir / fname, fname)

    all_done = all((dest_dir / f).exists() for _, f in parts)
    if all_done and not (dest_dir / 'CFAD').exists():
        print('  Extracting CFAD multi-part zip...')
        result = subprocess.run(['unzip', str(dest_dir / 'CFAD.zip'), '-d', str(dest_dir)])
        if result.returncode != 0:
            print('  [ERROR] Try manually: cd CFAD && unzip CFAD.zip')


# ═══════════════════════════════════════════════════════
# 6. CVoiceFake (~8 GB) — Zenodo 11229569
#    Voice cloning, 5 languages (EN, ZH, DE, FR, IT)
# ═══════════════════════════════════════════════════════
def download_cvoicefake():
    print('\n' + '='*60)
    print('6. CVoiceFake (~8 GB) — voice cloning, 5 languages')
    print('='*60)
    dest_dir = DATA_DIR / 'CVoiceFake'
    dest_dir.mkdir(exist_ok=True)

    # Check actual files on Zenodo record
    archive = dest_dir / 'CVoiceFake.zip'
    curl_download(
        'https://zenodo.org/records/11229569/files/CVoiceFake.zip?download=1',
        archive, 'CVoiceFake.zip'
    )
    if archive.exists() and archive.stat().st_size > 1e6:
        extract(archive, dest_dir)


# ═══════════════════════════════════════════════════════
# 7. DFADD (~5 GB) — HuggingFace
#    Diffusion & flow-matching based fakes (2024)
# ═══════════════════════════════════════════════════════
def download_dfadd():
    print('\n' + '='*60)
    print('7. DFADD (~5 GB) — diffusion/flow-matching fakes (2024)')
    print('='*60)
    dest_dir = DATA_DIR / 'DFADD'
    hf_download('isjwdu/DFADD', dest_dir)


# ═══════════════════════════════════════════════════════
# 8. MLAAD (~15 GB) — HuggingFace
#    51 languages, 140 TTS model architectures
# ═══════════════════════════════════════════════════════
def download_mlaad():
    print('\n' + '='*60)
    print('8. MLAAD (~15 GB) — 51 languages, 140 TTS models')
    print('='*60)
    dest_dir = DATA_DIR / 'MLAAD'
    hf_download('mueller91/MLAAD', dest_dir)


# ═══════════════════════════════════════════════════════
def print_summary():
    print('\n' + '='*60)
    print('DOWNLOAD SUMMARY')
    print('='*60)
    total = 0
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            total += size
            status = 'OK' if size > 1e8 else 'incomplete'
            print(f'  {d.name:<20} {size/1e9:6.2f} GB  [{status}]')
    print(f'  {"TOTAL":<20} {total/1e9:6.2f} GB')
    print(f'\n  Data dir: {DATA_DIR}')


if __name__ == '__main__':
    import sys
    print('DeepTruth — Audio Dataset Downloader (Full Suite)')
    print(f'Saving to: {DATA_DIR}')

    # Install huggingface_hub if needed
    try:
        import huggingface_hub
    except ImportError:
        print('Installing huggingface_hub...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'])

    download_wavefake()
    download_asvspoof2021()
    download_asvspoof2019()
    download_asvspoof5()
    download_cfad()
    download_cvoicefake()
    download_dfadd()
    download_mlaad()
    print_summary()
