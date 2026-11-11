#!/usr/bin/env python3
"""Check the frozen data, split, and summary hashes without rewriting results."""

from pathlib import Path
import hashlib
import json
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "results/RESULTS_MANIFEST.json").read_text())
    checksums = {
        "data/processed/pem_schema/cycpeptmpdb_pampa.jsonl": manifest["dataset_checksum"],
        **manifest["raw_data_checksums"],
        **manifest["split_checksums"],
        **manifest["artifact_checksums"],
    }
    failures = []
    for relative_path, expected in sorted(checksums.items()):
        path = ROOT / relative_path
        if not path.is_file():
            failures.append({"path": relative_path, "reason": "missing"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            failures.append({"path": relative_path, "reason": "checksum mismatch"})
    print(json.dumps({
        "status": "CHECKSUMS_MATCH" if not failures else "CHECKSUM_FAILURE",
        "checked_files": len(checksums),
        "archived_manifest_status": manifest["status"],
        "failures": failures,
    }, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
