#!/usr/bin/env python3
"""Seqwin integration test.

This script:
1. Downloads the test dataset `assemblies.tar`.
2. Extracts it into `assemblies/`.
3. Runs Seqwin with `targets.txt` and `non-targets.txt`.
4. Verifies the SHA-256 of `seqwin-out/signatures.fasta` against `expected-output/signatures.fasta`.

To run the script:
```bash
python run_test.py
```

To force a fresh download of the archive:
```bash
python run_test.py --force-download
```
"""

from __future__ import annotations

import os
import argparse
import hashlib
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

DATASET_URL = "https://github.com/treangenlab/Seqwin/releases/download/v0.1.0/assemblies.tar"
DATASET_SHA256 = "149cf4450b3877ab88913ab340fbee60fb12f23bc0f858746b37fb678ec7fca6"
CHUNK_SIZE = 1024 * 1024  # 1 MiB
THREADS = max(1, min(os.cpu_count() or 1, 4))


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_text(path_a, path_b, encoding="utf-8"):
    # ensure universal newline handling with newline=None
    with open(Path(path_a), "r", encoding=encoding, newline=None) as f1, \
         open(Path(path_b), "r", encoding=encoding, newline=None) as f2:
        return f1.read() == f2.read()


def download_file(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Using existing archive: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as output:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total is not None else None
            downloaded = 0
            next_report = 50 * CHUNK_SIZE

            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report or (total_bytes is not None and downloaded == total_bytes):
                    if total_bytes:
                        pct = downloaded / total_bytes * 100
                        print(
                            f"  downloaded {downloaded / CHUNK_SIZE:.1f} / "
                            f"{total_bytes / CHUNK_SIZE:.1f} MiB ({pct:.1f}%)"
                        )
                    else:
                        print(f"  downloaded {downloaded / CHUNK_SIZE:.1f} MiB")
                    next_report += 50 * CHUNK_SIZE

        tmp_path.replace(destination)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def safe_extract(tar_path: Path, destination: Path) -> None:
    extract_root = destination.parent
    if destination.exists():
        shutil.rmtree(destination)

    extract_root.mkdir(parents=True, exist_ok=True)
    extract_root_resolved = extract_root.resolve()

    with tarfile.open(tar_path, mode="r") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (extract_root / member.name).resolve()
            try:
                member_path.relative_to(extract_root_resolved)
            except ValueError:
                raise RuntimeError(f"Unsafe path detected in tar archive: {member.name}")
        archive.extractall(extract_root)


def run_seqwin(test_dir: Path) -> None:
    output_dir = test_dir / "seqwin-out"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    command = [
        "seqwin",
        "--tar-paths",
        "targets.txt",
        "--neg-paths",
        "non-targets.txt",
        "--no-mash",
        "--no-blast",
        "--threads",
        str(THREADS),
    ]

    print("Running:")
    print("  " + " ".join(command))
    subprocess.run(command, cwd=test_dir, check=True)


def verify_expected_output(test_dir: Path) -> None:
    actual = test_dir / "seqwin-out" / "signatures.fasta"
    expected = test_dir / "expected-output" / "signatures.fasta"

    if not expected.exists():
        raise FileNotFoundError(f"Expected output file not found: {expected}")
    if not actual.exists():
        raise FileNotFoundError(f"Actual output file not found: {actual}")

    if not same_text(actual, expected):
        raise SystemExit(
            "Integration test failed: seqwin-out/signatures.fasta does not match "
            "expected-output/signatures.fasta"
        )

    print("Integration test passed.")


def validate_inputs(test_dir: Path) -> None:
    required = [
        test_dir / "targets.txt",
        test_dir / "non-targets.txt",
        test_dir / "expected-output" / "signatures.fasta",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n  - ".join([""] + missing)
        raise FileNotFoundError(f"Missing required test files:{formatted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Seqwin release dataset integration test.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download assemblies.tar even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_dir = Path(__file__).resolve().parent
    archive_path = test_dir / "assemblies.tar"
    assemblies_dir = test_dir / "assemblies"

    validate_inputs(test_dir)
    download_file(DATASET_URL, archive_path, force=args.force_download)
    actual = sha256sum(archive_path)
    if actual != DATASET_SHA256:
        raise RuntimeError(
            f"Downloaded archive checksum mismatch:\n"
            f"expected: {DATASET_SHA256}\n"
            f"actual:   {actual}\n"
            f"Please delete the archive or rerun with --force-download."
        )
    safe_extract(archive_path, assemblies_dir)
    run_seqwin(test_dir)
    verify_expected_output(test_dir)


if __name__ == "__main__":
    main()
