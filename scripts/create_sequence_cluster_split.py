#!/usr/bin/env python3
"""Create a strict 70% sequence-identity connected-component split.

Sequence identity is the number of matches in an optimal global ``globalxx``
alignment divided by the longer sequence length. An undirected graph connects
every pair at or above the threshold. Connected components, rather than
representative-based greedy clusters, are assigned intact so that every
cross-partition sequence pair is below the threshold.

For this derivative-rich dataset the threshold graph contains one dominant
component. The assignment therefore prioritizes the isolation guarantee over
nominal 70/15/15 proportions and records the resulting imbalance explicitly.
"""

from collections import Counter, defaultdict
import csv
import json
import logging
from pathlib import Path
import sys

from Bio.Align import PairwiseAligner
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.serialization import load_jsonl, save_jsonl


IDENTITY_THRESHOLD = 0.70


def make_identity_aligner():
    """Return an aligner equivalent to Biopython ``globalxx`` scoring."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    return aligner


def sequence_identity(seq_a, seq_b, aligner=None):
    """Optimal-global-alignment matches divided by the longer sequence length."""
    if seq_a == seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    aligner = aligner or make_identity_aligner()
    return float(aligner.score(seq_a, seq_b)) / max(len(seq_a), len(seq_b))


def build_similarity_components(sequences, identity_threshold=IDENTITY_THRESHOLD):
    """Return threshold-graph connected components and graph diagnostics."""
    sequences = list(sequences)
    n_sequences = len(sequences)
    parent = list(range(n_sequences))
    component_size = [1] * n_sequences

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if component_size[left_root] < component_size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_size[left_root] += component_size[right_root]

    aligner = make_identity_aligner()
    edges = 0
    comparisons = 0
    for left, seq_left in enumerate(sequences):
        for right in range(left):
            seq_right = sequences[right]
            if min(len(seq_left), len(seq_right)) / max(len(seq_left), len(seq_right)) < identity_threshold:
                continue
            comparisons += 1
            if sequence_identity(seq_left, seq_right, aligner) >= identity_threshold:
                union(left, right)
                edges += 1

    members = defaultdict(list)
    for index in range(n_sequences):
        members[find(index)].append(index)
    components = sorted(members.values(), key=lambda group: sequences[min(group)])
    return components, {"candidate_comparisons": comparisons, "threshold_edges": edges}


def _balanced_two_way_partition(
    component_ids, component_weights, component_sequence_counts, component_label_sums
):
    """Divide components while balancing samples, labels, and sequence counts."""
    ordered = sorted(component_ids, key=lambda cid: (-component_weights[cid], cid))
    measures = [
        np.asarray([component_weights[cid] for cid in ordered], dtype=float),
        np.asarray([component_label_sums[cid] for cid in ordered], dtype=float),
        np.asarray([component_sequence_counts[cid] for cid in ordered], dtype=float),
        np.ones(len(ordered), dtype=float),
    ]
    totals = [float(values.sum()) for values in measures]
    n_binary = len(ordered)
    n_variables = n_binary + len(measures)
    objective = np.zeros(n_variables)
    objective[n_binary:] = [
        1_000_000 / max(totals[0], 1.0),
        10 / max(abs(totals[1]), 1.0),
        1 / max(totals[2], 1.0),
        0.1 / max(totals[3], 1.0),
    ]
    rows, upper = [], []
    for measure_index, (values, total) in enumerate(zip(measures, totals)):
        deviation_index = n_binary + measure_index
        positive = np.zeros(n_variables)
        positive[:n_binary] = values
        positive[deviation_index] = -1.0
        rows.append(positive)
        upper.append(total / 2)
        negative = np.zeros(n_variables)
        negative[:n_binary] = -values
        negative[deviation_index] = -1.0
        rows.append(negative)
        upper.append(-total / 2)
    result = milp(
        c=objective,
        integrality=np.r_[np.ones(n_binary), np.zeros(len(measures))],
        bounds=Bounds(
            np.zeros(n_variables),
            np.r_[np.ones(n_binary), np.full(len(measures), np.inf)],
        ),
        constraints=LinearConstraint(np.asarray(rows), -np.inf, np.asarray(upper)),
        options={"time_limit": 30.0},
    )
    if not result.success:
        raise RuntimeError(f"Unable to balance isolated components: {result.message}")
    left = {cid for cid, selected in zip(ordered, result.x[:n_binary]) if selected >= 0.5}
    return sorted(left), sorted(set(ordered) - left)


def assign_components(
    component_weights,
    component_sequence_counts,
    component_label_sums,
    target_fractions=(0.70, 0.15, 0.15),
):
    """Assign intact components while minimizing unavoidable size imbalance."""
    total = sum(component_weights.values())
    largest = max(component_weights, key=lambda cid: (component_weights[cid], -cid))
    targets = dict(zip(("train", "val", "test"), target_fractions))

    if component_weights[largest] > total * targets["train"]:
        val, test = _balanced_two_way_partition(
            [cid for cid in component_weights if cid != largest],
            component_weights,
            component_sequence_counts,
            component_label_sums,
        )
        return {"train": [largest], "val": val, "test": test}, True

    assignments = {name: [] for name in targets}
    sizes = Counter()
    for cid in sorted(component_weights, key=lambda key: (-component_weights[key], key)):
        destination = min(
            targets,
            key=lambda name: (
                abs((sizes[name] + component_weights[cid]) - total * targets[name])
                - abs(sizes[name] - total * targets[name]),
                sizes[name] / targets[name],
                name,
            ),
        )
        assignments[destination].append(cid)
        sizes[destination] += component_weights[cid]
    return assignments, False


def validate_cross_partition_identity(split_sequences, identity_threshold=IDENTITY_THRESHOLD):
    """Exhaustively validate and summarize all cross-partition sequence pairs."""
    aligner = make_identity_aligner()
    maxima = {}
    names = ("train", "val", "test")
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            maximum = 0.0
            for seq_left in split_sequences[left_name]:
                for seq_right in split_sequences[right_name]:
                    identity = sequence_identity(seq_left, seq_right, aligner)
                    maximum = max(maximum, identity)
                    if identity >= identity_threshold:
                        raise RuntimeError(
                            "Cross-partition identity violation: "
                            f"{left_name} {seq_left} vs {right_name} {seq_right} = {identity:.6f}"
                        )
            maxima[f"{left_name}_vs_{right_name}"] = maximum
    return maxima


def main():
    input_file = PROJECT_ROOT / "data/processed/pem_schema/cycpeptmpdb_pampa.jsonl"
    all_samples = load_jsonl(input_file)
    seq_to_samples = defaultdict(list)
    for sample in all_samples:
        seq_to_samples[sample.sequence].append(sample)
    unique_sequences = sorted(seq_to_samples)
    logger.info("Loaded %d samples and %d exact sequences", len(all_samples), len(unique_sequences))

    components, graph = build_similarity_components(unique_sequences)
    sequence_component = {}
    component_weights = {}
    component_sequence_counts = {}
    component_label_sums = {}
    for component_id, sequence_indices in enumerate(components):
        component_weights[component_id] = sum(
            len(seq_to_samples[unique_sequences[index]]) for index in sequence_indices
        )
        component_sequence_counts[component_id] = len(sequence_indices)
        component_label_sums[component_id] = sum(
            sample.label
            for index in sequence_indices
            for sample in seq_to_samples[unique_sequences[index]]
        )
        for index in sequence_indices:
            sequence_component[unique_sequences[index]] = component_id

    assignments, target_impossible = assign_components(
        component_weights, component_sequence_counts, component_label_sums
    )
    component_split = {
        component_id: split
        for split, component_ids in assignments.items()
        for component_id in component_ids
    }
    split_samples = {name: [] for name in ("train", "val", "test")}
    for sample in all_samples:
        split_samples[component_split[sequence_component[sample.sequence]]].append(sample)
    for samples in split_samples.values():
        samples.sort(key=lambda sample: sample.sample_id)

    split_sequences = {
        name: sorted({sample.sequence for sample in samples})
        for name, samples in split_samples.items()
    }
    cross_partition_maxima = validate_cross_partition_identity(split_sequences)

    output_dir = PROJECT_ROOT / "data/splits/CycPeptMPDB_PAMPA/sequence_cluster"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in split_samples.items():
        save_jsonl(samples, output_dir / f"{name}.jsonl")

    with (output_dir / "sequence_component_assignments.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "sequence", "component_id", "split"])
        for sample in all_samples:
            cid = sequence_component[sample.sequence]
            writer.writerow([sample.sample_id, sample.sequence, cid, component_split[cid]])

    largest_component = max(component_weights, key=component_weights.get)
    metadata = {
        "strategy": "strict_sequence_identity_connected_components",
        "identity_definition": "optimal globalxx alignment matches / longer sequence length",
        "identity_threshold": IDENTITY_THRESHOLD,
        "threshold_comparison": ">=",
        "n_total": len(all_samples),
        "n_unique_sequences": len(unique_sequences),
        "n_components": len(components),
        "threshold_edges": graph["threshold_edges"],
        "candidate_comparisons": graph["candidate_comparisons"],
        "largest_component_samples": component_weights[largest_component],
        "largest_component_unique_sequences": len(components[largest_component]),
        "nominal_70_15_15_impossible": target_impossible,
        "assignment_policy": "largest component to train; remaining components balanced by samples, labels, unique sequences, and component count"
        if target_impossible
        else "intact-component greedy assignment toward 70/15/15",
        "cross_partition_max_identity": cross_partition_maxima,
        "splits": {
            name: {
                "samples": len(split_samples[name]),
                "unique_sequences": len(split_sequences[name]),
                "components": len(assignments[name]),
            }
            for name in ("train", "val", "test")
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Strict split written: %s", json.dumps(metadata["splits"]))
    logger.warning(
        "Nominal 70/15/15 feasible: %s; largest component: %d/%d samples",
        not target_impossible,
        metadata["largest_component_samples"],
        len(all_samples),
    )


if __name__ == "__main__":
    main()
