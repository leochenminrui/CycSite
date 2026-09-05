#!/usr/bin/env python3
"""Count non-finite values in the realized A and RDKit 2D feature systems."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.data.serialization import load_samples
from src.benchmark.featurizers import AnchorAwareWrapper, RDKitFullFeaturizer

OUT = ROOT / "results/audits/descriptor_finiteness.csv"


def summarize(system, matrix, names):
    rows = []
    for index, name in enumerate(names):
        column = matrix[:, index]
        rows.append(
            {
                "system": system,
                "feature": name,
                "n_values": len(column),
                "n_nan": int(np.isnan(column).sum()),
                "n_positive_infinity": int(np.isposinf(column).sum()),
                "n_negative_infinity": int(np.isneginf(column).sum()),
                "n_nonfinite": int((~np.isfinite(column)).sum()),
            }
        )
    return rows


def main():
    samples = load_samples(ROOT / "data/processed/pem_schema/cycpeptmpdb_pampa.jsonl")
    chemistry = AnchorAwareWrapper(descriptor_set="basic", ablation_mode="chemistry_only")
    chemistry.fit(samples)
    a_matrix = chemistry.transform(samples)
    rdkit = RDKitFullFeaturizer()
    rdkit._desc_names = rdkit._get_all_descriptor_names()
    raw_rdkit = np.vstack([
        rdkit._compute_descriptors((sample.assay_metadata or {}).get("smiles"))
        for sample in samples
    ])
    rdkit.fit(samples)
    rdkit_matrix = rdkit.transform(samples)
    rows = summarize("Chemistry (A)", a_matrix, chemistry.get_feature_names())
    rows += summarize("RDKit 2D before imputation", raw_rdkit, rdkit.get_feature_names())
    rows += summarize("RDKit 2D after train-fitted median imputation", rdkit_matrix, rdkit.get_feature_names())
    output = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    print(output.groupby("system")[["n_nan", "n_positive_infinity", "n_negative_infinity", "n_nonfinite"]].sum())


if __name__ == "__main__":
    main()
