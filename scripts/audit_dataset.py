#!/usr/bin/env python3
"""Reviewer-requested peptide-length, SMILES, and descriptor inventories."""
from pathlib import Path
from collections import Counter
import ast
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.data.serialization import load_samples
from src.data.converters.cycpeptmpdb_converter import CycPeptMPDBConverter

OUT = ROOT / "results/audits"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    samples = load_samples(ROOT / "data/processed/pem_schema/cycpeptmpdb_pampa.jsonl")
    raw = pd.read_csv(
        ROOT / "data/raw/CycPeptMPDB_Peptide_Assay_PAMPA.csv", low_memory=False
    )
    lengths = [len(sample.sequence) for sample in samples]
    distribution = Counter(lengths)
    pd.DataFrame(
        [
            {"peptide_length": length, "sample_count": count, "fraction": count / len(samples)}
            for length, count in sorted(distribution.items())
        ]
    ).to_csv(OUT / "peptide_length_distribution.csv", index=False)

    smiles_rows = []
    for sample in samples:
        smiles = (sample.assay_metadata or {}).get("smiles")
        status = "missing" if not smiles else ("valid" if Chem.MolFromSmiles(smiles) is not None else "invalid")
        smiles_rows.append({"sample_id": sample.sample_id, "smiles_status": status})
    smiles = pd.DataFrame(smiles_rows)
    smiles.to_csv(OUT / "smiles_validity.csv", index=False)
    smiles.smiles_status.value_counts().rename_axis("status").reset_index(name="count").to_csv(
        OUT / "smiles_validity_summary.csv", index=False
    )

    pd.DataFrame(
        {
            "rdkit_descriptor_index": range(1, len(Descriptors.descList) + 1),
            "rdkit_descriptor_name": [name for name, _ in Descriptors.descList],
        }
    ).to_csv(OUT / "rdkit_2d_descriptor_inventory.csv", index=False)

    retained_rows = {sample.provenance["source_row_index"] for sample in samples}
    converter = CycPeptMPDBConverter(strict_mode=False)
    monomer_occurrences = Counter()
    monomer_raw_records = Counter()
    monomer_retained_records = Counter()
    mono_record_rows = []
    for row_index, value in enumerate(raw.Sequence):
        try:
            monomers = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            monomers = []
        symbols = [str(monomer) for monomer in monomers]
        monomer_occurrences.update(symbols)
        monomer_raw_records.update(set(symbols))
        if row_index in retained_rows:
            monomer_retained_records.update(set(symbols))
        placeholders = sorted({symbol for symbol in symbols if symbol.startswith("Mono")})
        if placeholders:
            mono_record_rows.append(
                {
                    "source_row_index": row_index,
                    "sample_id": f"CYCPEPTMPDB_PAMPA_{row_index:06d}",
                    "retained_for_modeling": row_index in retained_rows,
                    "mono_placeholder_symbols": ";".join(placeholders),
                    "mono_placeholder_occurrences": sum(
                        symbol.startswith("Mono") for symbol in symbols
                    ),
                }
            )

    def handling_status(symbol):
        if symbol in converter.monomer_to_aa:
            return "backbone_mapped"
        if symbol.startswith("Mono"):
            return "undefined_Mono_placeholder_skipped"
        if symbol == "ac-" or symbol.startswith("-"):
            return "explicit_terminal_rule"
        if symbol.endswith("-") or symbol in {"glyco-", "deca-", "medl-"}:
            return "explicit_nonbackbone_rule"
        return "unmapped_other"

    monomer_rows = [
        {
            "monomer_symbol": symbol,
            "handling_status": handling_status(symbol),
            "occurrences": monomer_occurrences[symbol],
            "raw_record_count": monomer_raw_records[symbol],
            "retained_record_count": monomer_retained_records[symbol],
        }
        for symbol in sorted(monomer_occurrences)
    ]
    monomer_frame = pd.DataFrame(monomer_rows)
    monomer_frame.to_csv(OUT / "monomer_coverage.csv", index=False)
    mono_records = pd.DataFrame(mono_record_rows)
    mono_records.to_csv(OUT / "mono_placeholder_records.csv", index=False)

    mono_symbols = monomer_frame[monomer_frame.handling_status == "undefined_Mono_placeholder_skipped"]
    unmapped_other = monomer_frame[monomer_frame.handling_status == "unmapped_other"]

    pd.DataFrame(
        [
            {
                "n_samples": len(samples),
                "minimum_length": min(lengths),
                "maximum_length": max(lengths),
                "single_residue_peptides": distribution.get(1, 0),
                "missing_smiles": int((smiles.smiles_status == "missing").sum()),
                "invalid_smiles": int((smiles.smiles_status == "invalid").sum()),
                "rdkit_descriptor_count": len(Descriptors.descList),
                "observed_monomer_symbols": len(monomer_frame),
                "observed_backbone_mapped_symbols": int(
                    (monomer_frame.handling_status == "backbone_mapped").sum()
                ),
                "mono_placeholder_symbols": len(mono_symbols),
                "mono_placeholder_raw_records": len(mono_records),
                "mono_placeholder_retained_records": int(
                    mono_records.retained_for_modeling.sum()
                ),
                "mono_placeholder_retained_occurrences": int(
                    mono_records.loc[
                        mono_records.retained_for_modeling, "mono_placeholder_occurrences"
                    ].sum()
                ),
                "unmapped_other_symbols": len(unmapped_other),
                "unmapped_other_symbol_list": ";".join(unmapped_other.monomer_symbol),
            }
        ]
    ).to_csv(OUT / "reviewer_audit_summary.csv", index=False)
    print((OUT / "reviewer_audit_summary.csv").read_text())


if __name__ == "__main__":
    main()
