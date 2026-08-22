#!/usr/bin/env python3
"""Create, verify, or unpack the two large Paper A evidence ledgers.

GitHub stores these transport copies as deterministic ``.csv.gz`` files.  The
uncompressed SHA-256 values are recorded alongside them, so extraction is
lossless and auditable before the short verification pipeline is run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results" / "paper_a" / "physics_optimization"
ARCHIVED_LEDGERS = (
    "fem_unique_geometries_mesh24.csv",
    "fem_verified_terminal_fast.csv",
)
TRANSPORT_MANIFEST = EVIDENCE / "compressed_evidence_manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def create_archives() -> None:
    records: dict[str, dict[str, object]] = {}
    for filename in ARCHIVED_LEDGERS:
        source = EVIDENCE / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        raw = source.read_bytes()
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        archive = source.with_suffix(f"{source.suffix}.gz")
        atomic_write(archive, compressed)
        records[filename] = {
            "archive": archive.name,
            "uncompressed_sha256": sha256_bytes(raw),
            "uncompressed_bytes": len(raw),
            "archive_sha256": sha256_bytes(compressed),
            "archive_bytes": len(compressed),
        }
    payload = {
        "format_version": 1,
        "compression": "gzip level 9 with mtime=0",
        "files": records,
    }
    atomic_write(
        TRANSPORT_MANIFEST,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def verify_or_unpack() -> None:
    manifest = json.loads(TRANSPORT_MANIFEST.read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        archive = EVIDENCE / str(expected["archive"])
        compressed = archive.read_bytes()
        if sha256_bytes(compressed) != expected["archive_sha256"]:
            raise RuntimeError(f"Compressed SHA-256 mismatch: {archive}")
        raw = gzip.decompress(compressed)
        if sha256_bytes(raw) != expected["uncompressed_sha256"]:
            raise RuntimeError(f"Uncompressed SHA-256 mismatch: {archive}")
        target = EVIDENCE / filename
        if target.exists():
            if sha256_bytes(target.read_bytes()) != expected["uncompressed_sha256"]:
                raise RuntimeError(f"Existing ledger differs from archive: {target}")
        else:
            atomic_write(target, raw)
        print(f"Verified {filename}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create-archives",
        action="store_true",
        help="create deterministic transport archives from the raw CSV ledgers",
    )
    args = parser.parse_args()
    if args.create_archives:
        create_archives()
    verify_or_unpack()


if __name__ == "__main__":
    main()
