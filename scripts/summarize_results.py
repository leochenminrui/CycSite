#!/usr/bin/env python3
"""Write a concise machine-readable summary of the analyses."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def main():
    summary = {
        "audit": records(
            pd.read_csv(RESULTS / "audits/reviewer_audit_summary.csv")
        )[0],
        "primary": json.loads((RESULTS / "key_results.json").read_text()),
    }

    shap_groups = pd.read_csv(RESULTS / "shap/group_summary_with_ci.csv")
    shap_features = pd.read_csv(RESULTS / "shap/feature_summary_with_ci.csv")
    shap_stability = pd.read_csv(RESULTS / "shap/pairwise_seed_stability.csv")
    full_stability = shap_stability[
        shap_stability.representation == "A+B+C"
    ][["feature_rank_spearman", "top10_jaccard"]]
    summary["shap"] = {
        "complete_representation_groups": records(
            shap_groups[
                (shap_groups.representation == "A+B+C")
                & shap_groups.group.isin(["A", "B", "C"])
            ]
        ),
        "complete_representation_top10_features": records(
            shap_features[shap_features.representation == "A+B+C"]
            .nlargest(10, "mean")
        ),
        "complete_representation_stability": {
            column: {
                "mean": float(full_stability[column].mean()),
                "median": float(full_stability[column].median()),
                "minimum": float(full_stability[column].min()),
                "maximum": float(full_stability[column].max()),
            }
            for column in full_stability
        },
    }

    time_performance = pd.read_csv(
        RESULTS / "time_forward/model_performance_with_ci.csv"
    )
    time_spearman = time_performance[time_performance.metric == "spearman"]
    time_associations = pd.read_csv(
        RESULTS / "time_forward/cutoff_associations.csv"
    )
    summary["time_forward"] = {
        "mean_spearman_across_cutoffs": records(
            time_spearman.groupby("model", as_index=False).estimate.mean()
            .sort_values("estimate", ascending=False)
        ),
        "complete_model_similarity_association": records(
            time_associations.query(
                "model == 'Chemistry + Site + Context' and diagnostic == 'median_max_tanimoto'"
            )
        )[0],
    }

    scaffold_summary = pd.read_csv(
        RESULTS / "scaffold_ranking/summary_with_ci.csv"
    )
    scaffold_paired = pd.read_csv(
        RESULTS / "scaffold_ranking/paired_comparison_AB_vs_A.csv"
    )
    summary["scaffold_ranking"] = {
        "models": records(scaffold_summary),
        "chemistry_site_minus_chemistry": records(scaffold_paired)[0],
    }

    summary["secondary_cross_estimator_analysis"] = records(
        pd.read_csv(
            RESULTS
            / "secondary_incremental_analysis/cross_estimator_paired_effects.csv"
        )
    )
    summary["exploratory_shap_performance_sensitivity"] = records(
        pd.read_csv(
            RESULTS
            / "secondary_incremental_analysis/shap_performance_associations.csv"
        )
    )

    output = RESULTS / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
