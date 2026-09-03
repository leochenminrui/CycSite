#!/usr/bin/env python3
"""Run the 30-seed primary benchmark with resumable bounded parallelism."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import sys
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.benchmark.runner import run_single_model

MODELS = [
    "seq_aa_xgb",
    "chem_A_xgb",
    "site_B_xgb",
    "context_C_xgb",
    "chem_site_AB_xgb",
    "chem_context_AC_xgb",
    "site_context_BC_xgb",
    "full_ABC_xgb",
    "rdkit_full_xgb",
    "ecfp_xgb",
]
SPLITS = ["random", "sequence_cluster"]
SEEDS = range(30)
OUT = ROOT / "results/raw_runs"


def valid(model_id, split, seed):
    base = OUT / split / f"seed_{seed}" / model_id
    metrics = base / "metrics.json"
    trials = base / "optuna_trials.csv"
    predictions = base / "predictions.csv"
    if not (metrics.exists() and trials.exists() and predictions.exists()):
        return False
    try:
        payload = json.loads(metrics.read_text())
        split_dir = ROOT / "data/splits/CycPeptMPDB_PAMPA" / split
        expected_sizes = {
            name: sum(1 for _ in (split_dir / f"{name}.jsonl").open())
            for name in ("train", "val", "test")
        }
        expected_test_ids = [
            json.loads(line)["sample_id"]
            for line in (split_dir / "test.jsonl").read_text().splitlines()
        ]
        prediction_ids = pd.read_csv(predictions, usecols=["sample_id"]).sample_id.tolist()
        return (
            payload.get("status") == "completed"
            and payload.get("n_optuna_trials") == 50
            and sum(1 for _ in trials.open()) - 1 == 50
            and payload.get("n_train") == expected_sizes["train"]
            and payload.get("n_val") == expected_sizes["val"]
            and payload.get("n_test") == expected_sizes["test"]
            and prediction_ids == expected_test_ids
        )
    except Exception:
        return False


def one(task):
    model_id, split, seed, config = task
    result = run_single_model(model_id, split, seed, config, OUT, resume=False)
    return model_id, split, seed, result


def main():
    config = yaml.safe_load(
        (ROOT / "configs/benchmark/benchmark.yaml").read_text()
    )
    tasks = [
        (model_id, split, seed, config)
        for split in SPLITS
        for model_id in MODELS
        for seed in SEEDS
        if not valid(model_id, split, seed)
    ]
    workers = min(int(os.environ.get("CASPER_WORKERS", "8")), 8)
    print(json.dumps({"missing_runs": len(tasks), "workers": workers}), flush=True)
    completed = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, task): task[:3] for task in tasks}
        for future in as_completed(futures):
            key = futures[future]
            try:
                model_id, split, seed, result = future.result()
                ok = result.get("status") == "completed"
                completed += int(ok)
                failed += int(not ok)
                print(
                    json.dumps(
                        {
                            "model": model_id,
                            "split": split,
                            "seed": seed,
                            "status": result.get("status"),
                            "r2": result.get("test_metrics", {}).get("r2"),
                            "completed": completed,
                            "failed": failed,
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                print(json.dumps({"task": key, "status": "failed", "error": repr(exc)}), flush=True)
    print(json.dumps({"completed": completed, "failed": failed, "expected": len(tasks)}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
