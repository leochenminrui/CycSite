#!/usr/bin/env python3
"""Run the limited 30-seed position-summary ablation on the fixed random split."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.benchmark.runner import run_single_model

MODEL = "chem_site_no_position_summary_xgb"
SPLIT = "random"
OUT = ROOT / "results/position_summary_ablation/raw_runs"


def valid(seed):
    base = OUT / SPLIT / f"seed_{seed}" / MODEL
    metrics = base / "metrics.json"
    trials = base / "optuna_trials.csv"
    predictions = base / "predictions.csv"
    if not (metrics.exists() and trials.exists() and predictions.exists()):
        return False
    try:
        payload = json.loads(metrics.read_text())
        return (
            payload.get("status") == "completed"
            and payload.get("n_optuna_trials") == 50
            and payload.get("n_features") == 42
            and sum(1 for _ in trials.open()) - 1 == 50
            and len(pd.read_csv(predictions)) == 1084
        )
    except Exception:
        return False


def one(task):
    seed, config = task
    result = run_single_model(MODEL, SPLIT, seed, config, OUT, resume=False)
    return seed, result


def main():
    config = yaml.safe_load((ROOT / "configs/benchmark/benchmark.yaml").read_text())
    tasks = [(seed, config) for seed in range(30) if not valid(seed)]
    workers = min(int(os.environ.get("CASPER_WORKERS", "8")), 8)
    print(json.dumps({"missing_runs": len(tasks), "workers": workers}), flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, task): task[0] for task in tasks}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                _, result = future.result()
                print(json.dumps({"seed": seed, "status": result.get("status"), "r2": result.get("test_metrics", {}).get("r2")}), flush=True)
                failed += int(result.get("status") != "completed")
            except Exception as exc:
                failed += 1
                print(json.dumps({"seed": seed, "status": "failed", "error": repr(exc)}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
