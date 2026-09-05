#!/usr/bin/env python3
"""Grouped paired bootstrap for the A+B versus A primary comparison.

The resampling unit is an exact peptide sequence. Each bootstrap replicate
draws sequence groups once and applies the identical sampled record indices to
both representations and all 30 matched optimization seeds. The analysis
therefore measures sensitivity to the composition of the fixed held-out test
set given the already trained models; it does not resample training data.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.data.serialization import load_samples

RESULTS = ROOT / "results"
OUT = RESULTS / "grouped_paired_bootstrap"
N_BOOTSTRAP = 5000
BASE_SEED = 20260905
MODELS = {"A": "chem_A_xgb", "A+B": "chem_site_AB_xgb"}


def load_prediction_matrix(split):
    frames = {}
    reference_ids = reference_y = None
    for label, model_id in MODELS.items():
        predictions = []
        for seed in range(30):
            path = RESULTS / "raw_runs" / split / f"seed_{seed}" / model_id / "predictions.csv"
            frame = pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)
            ids = frame.sample_id.to_numpy()
            y = frame.y_true.to_numpy(dtype=float)
            if reference_ids is None:
                reference_ids, reference_y = ids, y
            if not np.array_equal(ids, reference_ids) or not np.allclose(y, reference_y):
                raise ValueError(f"Prediction alignment mismatch: {path}")
            predictions.append(frame.y_pred.to_numpy(dtype=float))
        frames[label] = np.vstack(predictions)
    return reference_ids, reference_y, frames


def group_indices(split, sample_ids):
    samples = load_samples(ROOT / "data/splits/CycPeptMPDB_PAMPA" / split / "test.jsonl")
    sequence_by_id = {sample.sample_id: sample.sequence for sample in samples}
    missing = sorted(set(sample_ids) - set(sequence_by_id))
    if missing:
        raise ValueError(f"Missing sequence groups for {len(missing)} test records")
    groups = pd.Series(sample_ids).map(sequence_by_id).to_numpy()
    unique_groups = pd.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique_groups]
    if sum(map(len, indices)) != len(sample_ids):
        raise AssertionError("Sequence groups do not cover the complete test set")
    return unique_groups, indices


def metric_differences(y, pred_a, pred_ab, indices):
    yt = y[indices]
    pa = pred_a[:, indices]
    pab = pred_ab[:, indices]
    denominator = np.sum((yt - yt.mean()) ** 2)
    if denominator <= 0:
        return None
    r2_difference = np.mean(
        (np.sum((yt - pa) ** 2, axis=1) - np.sum((yt - pab) ** 2, axis=1))
        / denominator
    )
    rmse_difference = np.mean(
        np.sqrt(np.mean((yt - pab) ** 2, axis=1))
        - np.sqrt(np.mean((yt - pa) ** 2, axis=1))
    )
    mae_difference = np.mean(
        np.mean(np.abs(yt - pab), axis=1) - np.mean(np.abs(yt - pa), axis=1)
    )
    return {"r2": r2_difference, "rmse": rmse_difference, "mae": mae_difference}


def analyze_split(split, split_index):
    sample_ids, y, predictions = load_prediction_matrix(split)
    groups, members = group_indices(split, sample_ids)
    observed = metric_differences(
        y, predictions["A"], predictions["A+B"], np.arange(len(y))
    )
    rng_seed = BASE_SEED + split_index
    rng = np.random.default_rng(rng_seed)
    bootstrap_rows = []
    invalid = 0
    for replicate in range(N_BOOTSTRAP):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([members[index] for index in selected])
        values = metric_differences(y, predictions["A"], predictions["A+B"], indices)
        if values is None:
            invalid += 1
            continue
        for metric, value in values.items():
            bootstrap_rows.append(
                {"split": split, "replicate": replicate, "metric": metric, "difference": value}
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    summaries = []
    for metric, observed_value in observed.items():
        values = bootstrap.loc[bootstrap.metric == metric, "difference"].to_numpy()
        summaries.append(
            {
                "split": split,
                "comparison": "A+B minus A",
                "metric": metric,
                "observed_mean_across_30_matched_seeds": observed_value,
                "grouped_bootstrap_ci95_low": np.quantile(values, 0.025),
                "grouped_bootstrap_ci95_high": np.quantile(values, 0.975),
                "n_test_records": len(sample_ids),
                "n_sequence_groups": len(groups),
                "bootstrap_replicates_requested": N_BOOTSTRAP,
                "bootstrap_replicates_valid": len(values),
                "bootstrap_replicates_invalid_zero_variance": invalid,
                "bootstrap_seed": rng_seed,
                "bootstrap_unit": "exact peptide sequence",
                "interval": "percentile",
            }
        )
    return pd.DataFrame(summaries), bootstrap


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    replicates = []
    for index, split in enumerate(("random", "sequence_cluster")):
        summary, bootstrap = analyze_split(split, index)
        summaries.append(summary)
        replicates.append(bootstrap)
    summary = pd.concat(summaries, ignore_index=True)
    samples = pd.concat(replicates, ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    samples.to_csv(OUT / "bootstrap_replicates.csv", index=False)
    metadata = {
        "analysis_type": "fixed-model held-out test-composition sensitivity",
        "comparison": "A+B minus A",
        "trained_models": "30 matched XGBoost optimization seeds per representation",
        "bootstrap_unit": "exact peptide sequence",
        "shared_resampling": (
            "Within each replicate, identical sampled record indices are used for A, A+B, "
            "and all 30 matched seeds."
        ),
        "limitations": [
            "Training and validation data are fixed and are not resampled.",
            "The analysis does not remove related derivatives from the original random split.",
            "The analysis is not an external validation or an uncertainty estimate for retraining on a new dataset.",
        ],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
