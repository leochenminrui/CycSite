#!/usr/bin/env python3
"""Rerun the key descriptor contrasts after excluding all Mono records."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.benchmark.runner import run_single_model
from scripts.compute_statistics import bootstrap_interval, monte_carlo_signflip

RESULTS = ROOT / "results/mono_exclusion_sensitivity"
MODELS = {
    "A": "chem_A_xgb",
    "A+B": "chem_site_AB_xgb",
    "A+B+C": "full_ABC_xgb",
}
SPLITS = ("random_no_mono", "sequence_cluster_no_mono")


def one(task):
    model_id, split, seed, config = task
    return model_id, split, seed, run_single_model(
        model_id, split, seed, config, RESULTS / "raw_runs", resume=True
    )


def valid(model_id, split, seed):
    path = RESULTS / "raw_runs" / split / f"seed_{seed}" / model_id / "metrics.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        return payload.get("status") == "completed" and payload.get("n_optuna_trials") == 50
    except Exception:
        return False


def aggregate():
    rows = []
    for split in SPLITS:
        for representation, model_id in MODELS.items():
            for seed in range(30):
                path = RESULTS / "raw_runs" / split / f"seed_{seed}" / model_id / "metrics.json"
                payload = json.loads(path.read_text())
                rows.append(
                    {
                        "split": split,
                        "representation": representation,
                        "model_id": model_id,
                        "seed": seed,
                        "n_train": payload["n_train"],
                        "n_val": payload["n_val"],
                        "n_test": payload["n_test"],
                        **{name: payload["test_metrics"][name] for name in ("r2", "rmse", "mae", "spearman")},
                    }
                )
    seed_level = pd.DataFrame(rows)
    seed_level.to_csv(RESULTS / "seed_level_results.csv", index=False)

    summaries = []
    for (split, representation), frame in seed_level.groupby(["split", "representation"], sort=False):
        for metric_index, metric in enumerate(("r2", "rmse", "mae", "spearman")):
            values = frame.sort_values("seed")[metric].to_numpy()
            low, high = bootstrap_interval(values, 20269600 + metric_index)
            summaries.append(
                {
                    "split": split,
                    "representation": representation,
                    "metric": metric,
                    "mean": values.mean(),
                    "sample_sd": values.std(ddof=1),
                    "bootstrap_bca_ci95_low": low,
                    "bootstrap_bca_ci95_high": high,
                    "n_seeds": len(values),
                    "n_test": int(frame.n_test.iloc[0]),
                }
            )
    pd.DataFrame(summaries).to_csv(RESULTS / "summary_with_ci.csv", index=False)

    contrasts = []
    for split_index, split in enumerate(SPLITS):
        frame = seed_level[seed_level.split == split]
        for contrast_index, (left, right) in enumerate((("A+B", "A"), ("A+B+C", "A+B"))):
            left_values = frame[frame.representation == left].sort_values("seed").r2.to_numpy()
            right_values = frame[frame.representation == right].sort_values("seed").r2.to_numpy()
            differences = left_values - right_values
            low, high = bootstrap_interval(differences, 20269700 + 10 * split_index + contrast_index)
            contrasts.append(
                {
                    "split": split,
                    "contrast": f"{left} minus {right}",
                    "mean_difference_r2": differences.mean(),
                    "bootstrap_bca_ci95_low": low,
                    "bootstrap_bca_ci95_high": high,
                    "p_value_signflip_mc": monte_carlo_signflip(
                        differences, 20269800 + 10 * split_index + contrast_index
                    ),
                    "n_pairs": len(differences),
                }
            )
    pd.DataFrame(contrasts).to_csv(RESULTS / "paired_contrasts.csv", index=False)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/benchmark/benchmark.yaml").read_text())
    tasks = [
        (model_id, split, seed, config)
        for split in SPLITS
        for model_id in MODELS.values()
        for seed in range(30)
        if not valid(model_id, split, seed)
    ]
    workers = min(int(os.environ.get("CASPER_WORKERS", "8")), 8)
    print(json.dumps({"missing_runs": len(tasks), "workers": workers}), flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, task): task[:3] for task in tasks}
        completed = 0
        for future in as_completed(futures):
            key = futures[future]
            try:
                _, _, _, result = future.result()
                completed += 1
                failures += result.get("status") != "completed"
                print(json.dumps({"task": key, "status": result.get("status"), "completed": completed}), flush=True)
            except Exception as exc:
                failures += 1
                print(json.dumps({"task": key, "status": "failed", "error": repr(exc)}), flush=True)
    if failures:
        raise SystemExit(f"{failures} sensitivity runs failed")
    aggregate()


if __name__ == "__main__":
    main()
