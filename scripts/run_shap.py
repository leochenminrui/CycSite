#!/usr/bin/env python3
"""Thirty-seed TreeSHAP attribution and cross-seed stability diagnostics."""
from itertools import combinations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.benchmark.featurizers import FEATURIZER_REGISTRY
from src.data.serialization import load_samples
from baselines.featurizers.anchor_aware_descriptors import get_feature_block_mapping

RAW = ROOT / "results/raw_runs/random"
OUT = ROOT / "results/shap"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "A": ("chem_A_xgb", "anchor_aware", {"descriptor_set": "basic", "ablation_mode": "chemistry_only"}),
    "B": ("site_B_xgb", "site_only", {}),
    "A+B": ("chem_site_AB_xgb", "anchor_aware", {"descriptor_set": "basic", "ablation_mode": "chemistry_anchors"}),
    "A+B+C": ("full_ABC_xgb", "anchor_aware", {"descriptor_set": "basic", "ablation_mode": "full"}),
}

FEATURE_TO_BLOCK = get_feature_block_mapping()


def subblock(name):
    try:
        return FEATURE_TO_BLOCK[name]
    except KeyError as error:
        raise KeyError(f"Feature is absent from canonical block mapping: {name}") from error


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    if np.allclose(values, values[0], rtol=0.0, atol=1e-15):
        return mean, values.std(ddof=1), float(values[0]), float(values[0])
    low, high = stats.bootstrap(
        (values,), np.mean, n_resamples=100_000, method="BCa",
        rng=np.random.default_rng(20260902),
    ).confidence_interval
    return mean, values.std(ddof=1), low, high


def main():
    mapping_payload = {
        "schema": "frozen basic A/B1/B2/B3/C descriptor mapping",
        "feature_count": len(FEATURE_TO_BLOCK),
        "features": [
            {"feature": feature, "block": block}
            for feature, block in FEATURE_TO_BLOCK.items()
        ],
    }
    (RESULTS / "feature_to_block_mapping.json").write_text(
        json.dumps(mapping_payload, indent=2) + "\n"
    )
    split = ROOT / "data/splits/CycPeptMPDB_PAMPA/random"
    train = load_samples(split / "train.jsonl")
    test = load_samples(split / "test.jsonl")
    y_train = np.asarray([sample.label for sample in train])
    subset = np.random.default_rng(20260717).choice(
        len(test), min(500, len(test)), replace=False
    )
    group_rows = []
    feature_rows = []

    for representation, (model_id, featurizer_key, kwargs) in MODELS.items():
        featurizer = FEATURIZER_REGISTRY[featurizer_key](**kwargs)
        featurizer.fit(train)
        x_train = np.nan_to_num(featurizer.transform(train))
        x_test = np.nan_to_num(featurizer.transform(test))[subset]
        names = featurizer.get_feature_names()
        for seed in range(30):
            params = json.loads(
                (RAW / f"seed_{seed}" / model_id / "best_params.json").read_text()
            )
            model = XGBRegressor(
                **params, tree_method="hist", verbosity=0, n_jobs=2,
                random_state=seed,
            )
            model.fit(x_train, y_train, verbose=False)
            contributions = np.asarray(
                model.get_booster().predict(
                    xgb.DMatrix(x_test), pred_contribs=True
                )
            )[:, :-1]
            mean_absolute = np.abs(contributions).mean(axis=0)
            total = mean_absolute.sum()
            proportions = mean_absolute / total
            grouped = {key: 0.0 for key in ["A", "B1", "B2", "B3", "C"]}
            for name, value in zip(names, proportions):
                grouped[subblock(name)] += float(value)
                feature_rows.append(
                    {
                        "representation": representation,
                        "seed": seed,
                        "feature": name,
                        "subblock": subblock(name),
                        "proportion": float(value),
                    }
                )
            grouped["B"] = grouped["B1"] + grouped["B2"] + grouped["B3"]
            for group, value in grouped.items():
                group_rows.append(
                    {
                        "representation": representation,
                        "seed": seed,
                        "group": group,
                        "attribution_proportion": value,
                        "n_explained": len(subset),
                    }
                )

    features = pd.DataFrame(feature_rows)
    groups = pd.DataFrame(group_rows)
    features.to_csv(OUT / "seed_level_feature_attribution.csv", index=False)
    groups.to_csv(OUT / "seed_level_group_attribution.csv", index=False)

    summaries = []
    for (representation, group), frame in groups.groupby(["representation", "group"]):
        mean, sample_sd, low, high = interval(frame.attribution_proportion)
        summaries.append(
            {
                "representation": representation,
                "group": group,
                "mean": mean,
                "sample_sd": sample_sd,
                "bootstrap_bca_ci95_low": low,
                "bootstrap_bca_ci95_high": high,
                "n_seeds": len(frame),
            }
        )
    pd.DataFrame(summaries).to_csv(OUT / "group_summary_with_ci.csv", index=False)

    feature_summaries = []
    for (representation, feature, block), frame in features.groupby(
        ["representation", "feature", "subblock"]
    ):
        mean, sample_sd, low, high = interval(frame.proportion)
        feature_summaries.append(
            {
                "representation": representation,
                "feature": feature,
                "subblock": block,
                "mean": mean,
                "sample_sd": sample_sd,
                "bootstrap_bca_ci95_low": low,
                "bootstrap_bca_ci95_high": high,
                "n_seeds": len(frame),
            }
        )
    feature_summary = pd.DataFrame(feature_summaries)
    feature_summary.to_csv(OUT / "feature_summary_with_ci.csv", index=False)

    stability_rows = []
    for representation, frame in features.groupby("representation"):
        matrix = frame.pivot(index="seed", columns="feature", values="proportion").sort_index()
        top10 = {
            seed: set(matrix.loc[seed].nlargest(min(10, matrix.shape[1])).index)
            for seed in matrix.index
        }
        for left, right in combinations(matrix.index, 2):
            rho = stats.spearmanr(matrix.loc[left], matrix.loc[right]).statistic
            union = top10[left] | top10[right]
            jaccard = len(top10[left] & top10[right]) / len(union) if union else 1.0
            stability_rows.append(
                {
                    "representation": representation,
                    "seed_left": left,
                    "seed_right": right,
                    "feature_rank_spearman": rho,
                    "top10_jaccard": jaccard,
                }
            )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(OUT / "pairwise_seed_stability.csv", index=False)
    stability.groupby("representation")[["feature_rank_spearman", "top10_jaccard"]].agg(
        ["mean", "median", "min", "max"]
    ).to_csv(OUT / "stability_summary.csv")
    print("30-seed SHAP analysis complete")


if __name__ == "__main__":
    main()
