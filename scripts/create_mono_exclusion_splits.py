#!/usr/bin/env python3
"""Create sensitivity-analysis splits excluding every Mono-placeholder record."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.data.serialization import load_samples, save_jsonl


def main():
    audit = pd.read_csv(
        ROOT / "results/audits/mono_placeholder_records.csv"
    )
    excluded = set(audit.loc[audit.retained_for_modeling, "sample_id"])
    summary = {}
    base = ROOT / "data/splits/CycPeptMPDB_PAMPA"
    for source_name, destination_name in (
        ("random", "random_no_mono"),
        ("sequence_cluster", "sequence_cluster_no_mono"),
    ):
        destination = base / destination_name
        destination.mkdir(parents=True, exist_ok=True)
        summary[destination_name] = {}
        for split in ("train", "val", "test"):
            samples = load_samples(base / source_name / f"{split}.jsonl")
            retained = [sample for sample in samples if sample.sample_id not in excluded]
            save_jsonl(retained, destination / f"{split}.jsonl")
            summary[destination_name][split] = {
                "source_samples": len(samples),
                "retained_samples": len(retained),
                "excluded_samples": len(samples) - len(retained),
            }
        metadata = {
            "strategy": "Mono-placeholder exclusion sensitivity",
            "source_split": source_name,
            "exclusion_rule": "raw Sequence contains a token beginning with 'Mono'",
            "splits": summary[destination_name],
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
