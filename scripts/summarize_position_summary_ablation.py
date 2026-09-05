#!/usr/bin/env python3
"""Summarize the position-summary ablation against existing A and A+B runs."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.compute_statistics import bootstrap_interval, monte_carlo_signflip

RESULTS = ROOT / "results"
OUT = RESULTS / "position_summary_ablation"


def metric_payload(path):
    payload = json.loads(path.read_text())
    return payload["test_metrics"]


def main():
    rows = []
    locations = {
        "A": RESULTS / "raw_runs/random",
        "A+B": RESULTS / "raw_runs/random",
        "A+B without position summaries": OUT / "raw_runs/random",
    }
    model_ids = {
        "A": "chem_A_xgb",
        "A+B": "chem_site_AB_xgb",
        "A+B without position summaries": "chem_site_no_position_summary_xgb",
    }
    for representation, base in locations.items():
        for seed in range(30):
            payload = metric_payload(base / f"seed_{seed}" / model_ids[representation] / "metrics.json")
            rows.append({"representation": representation, "seed": seed, **{m: payload[m] for m in ("r2", "rmse", "mae")}})
    seed_level = pd.DataFrame(rows)
    seed_level.to_csv(OUT / "seed_level_metrics.csv", index=False)
    summaries = []
    for representation, frame in seed_level.groupby("representation", sort=False):
        for metric_index, metric in enumerate(("r2", "rmse", "mae")):
            values = frame.sort_values("seed")[metric].to_numpy()
            low, high = bootstrap_interval(values, 20264000 + 10 * list(locations).index(representation) + metric_index)
            summaries.append({"representation": representation, "metric": metric, "n_test": 1084, "n_seeds": 30, "mean": values.mean(), "bootstrap_bca_ci95_low": low, "bootstrap_bca_ci95_high": high})
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    contrasts = []
    for contrast_index, (left, right) in enumerate((("A+B without position summaries", "A"), ("A+B without position summaries", "A+B"))):
        for metric_index, metric in enumerate(("r2", "rmse", "mae")):
            l = seed_level[seed_level.representation == left].sort_values("seed")[metric].to_numpy()
            r = seed_level[seed_level.representation == right].sort_values("seed")[metric].to_numpy()
            differences = l - r
            low, high = bootstrap_interval(differences, 20265000 + 10 * contrast_index + metric_index)
            contrasts.append({"comparison": f"{left} minus {right}", "metric": metric, "mean_difference": differences.mean(), "bootstrap_bca_ci95_low": low, "bootstrap_bca_ci95_high": high, "p_value_signflip_mc": monte_carlo_signflip(differences, 20266000 + 10 * contrast_index + metric_index), "n_pairs": 30})
    pd.DataFrame(contrasts).to_csv(OUT / "paired_contrasts.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({"analysis_type": "position-summary ablation", "removed_features": ["anchor_pos_mean", "anchor_pos_std", "anchor_pos_range"], "retained_site_features": ["anchor counts", "anchor density", "anchor-residue identities", "anchor-residue property classes", "terminal-edit annotations"], "split": "fixed random", "optimization": "30 matched seeds and 50 Optuna trials selected by validation RMSE"}, indent=2) + "\n")
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    main()
