#!/usr/bin/env python3
"""Run the complete 30-seed, five-estimator by seven-representation matrix."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
OUT = ROOT / "results/estimator_matrix/raw_runs"
FEATURES = ["Chem", "Site", "Context", "Chem+Site", "Chem+Context", "Site+Context", "Chem+Site+Context"]
ESTIMATORS = ["ridge", "elasticnet", "random_forest", "svr", "xgboost"]


def safe_name(feature):
    return feature.replace(" ", "_").replace("+", "_")


def valid(feature, estimator, seed):
    base = OUT / f"{safe_name(feature)}_{estimator}" / f"seed_{seed}"
    metrics = base / "metrics.json"
    trials = base / "optuna_trials.csv"
    predictions = base / "predictions.csv"
    if not (metrics.exists() and trials.exists() and predictions.exists()):
        return False
    try:
        payload = json.loads(metrics.read_text())
        return payload.get("status") == "completed" and payload.get("n_trials") == 10 and len(__import__('pandas').read_csv(trials)) == 10
    except Exception:
        return False


def run(task):
    feature, estimator, seed = task
    from scripts import _estimator_matrix_worker as module
    module.OUTPUT_DIR = OUT
    module.N_TRIALS = 10
    result = module.run_one(estimator, feature, seed)
    return feature, estimator, seed, result


def main():
    tasks = [
        (feature, estimator, seed)
        for feature in FEATURES
        for estimator in ESTIMATORS
        for seed in range(30)
        if not valid(feature, estimator, seed)
    ]
    workers = min(int(os.environ.get("CASPER_ESTIMATOR_WORKERS", "8")), 8)
    print(json.dumps({"missing_runs": len(tasks), "workers": workers}), flush=True)
    completed = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                feature, estimator, seed, result = future.result()
                completed += 1
                print(json.dumps({"feature": feature, "estimator": estimator, "seed": seed, "r2": result["test_metrics"]["r2"], "completed": completed, "failed": failed}), flush=True)
            except Exception as exc:
                failed += 1
                print(json.dumps({"task": task, "status": "failed", "error": repr(exc)}), flush=True)
    print(json.dumps({"completed": completed, "failed": failed, "expected": len(tasks)}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
