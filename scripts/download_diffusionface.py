"""
Manual download of selected DiffusionFace files from Zenodo.
Downloads the 5 most unique diffusion model types not already covered by DeepFakeFace.
Supports resume via HTTP Range requests.
"""
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "diffusionface"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    # (filename, url, size_bytes)  — using direct download URLs, not API endpoints
    ("ADM.tar",      "https://zenodo.org/records/10865300/files/ADM.tar",      2_820_403_200),
    ("DDIM.tar",     "https://zenodo.org/records/10865300/files/DDIM.tar",     3_222_650_880),
    ("PNDM.tar",     "https://zenodo.org/records/10865300/files/PNDM.tar",     3_201_648_640),
    ("LDM.tar",      "https://zenodo.org/records/10865300/files/LDM.tar",      2_991_790_080),
    ("DiffSwap.tar", "https://zenodo.org/records/10865300/files/DiffSwap.tar", 3_529_318_400),
]


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_with_resume(url, dest: Path, total_size: int):
    import subprocess
    existing = dest.stat().st_size if dest.exists() else 0

    if existing >= total_size:
        print(f"  [SKIP] {dest.name} already complete ({human_size(existing)})")
        return True

    if existing > 0:
        print(f"  [RESUME] {dest.name} — resuming from {human_size(existing)}")
    else:
        print(f"  [START] {dest.name} — {human_size(total_size)}")

    # Use curl: -L follows redirects, -C - resumes, --retry 3 retries on failure
    cmd = [
        "curl", "-L", "-C", "-",
        "--retry", "3", "--retry-delay", "5",
        "--progress-bar",
        "-o", str(dest),
        url,
    ]
    try:
        result = subprocess.run(cmd, check=True)
        actual = dest.stat().st_size if dest.exists() else 0
        print(f"  Downloaded: {human_size(actual)}")
        return actual > 0
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] curl failed with code {e.returncode}")
        return False
    except FileNotFoundError:
        print("  [ERROR] curl not found — trying wget")
        cmd2 = ["wget", "-c", "--show-progress", "-O", str(dest), url]
        try:
            subprocess.run(cmd2, check=True)
            return dest.exists() and dest.stat().st_size > 0
        except Exception as e2:
            print(f"  [ERROR] wget also failed: {e2}")
            return False


def extract_tar(tar_path: Path, out_dir: Path):
    marker = out_dir / f".extracted_{tar_path.stem}"
    if marker.exists():
        print(f"  [SKIP] {tar_path.name} already extracted")
        return

    print(f"  [EXTRACT] {tar_path.name} -> {out_dir}/")
    with tarfile.open(tar_path) as tf:
        tf.extractall(out_dir)
    marker.touch()
    tar_path.unlink()  # free disk space after extraction
    print(f"  [OK] extracted and removed tar")


def main():
    total_gb = sum(s for _, _, s in FILES) / 1e9
    print(f"DiffusionFace selective download — {len(FILES)} files, ~{total_gb:.1f} GB")
    print(f"Output: {OUT_DIR}\n")

    for name, url, size in FILES:
        dest = OUT_DIR / name
        print(f"\n{'='*60}")
        print(f"  {name}  ({human_size(size)})")
        ok = download_with_resume(url, dest, size)
        if ok and dest.exists():
            extract_tar(dest, OUT_DIR / name.replace(".tar", ""))

    print("\n\nDone. Contents of diffusionface/:")
    for d in sorted(OUT_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            count = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  {d.name}/  ({count} files)")


if __name__ == "__main__":
    main()
