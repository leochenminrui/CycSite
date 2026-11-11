# CycSite

**Site-conditioned chemical edits for cyclic-peptide permeability modeling.**

CycSite studies how much the location of a chemical edit adds to the prediction
of cyclic-peptide passive membrane permeability. It separates whole-molecule
chemistry, edit-site descriptors, and multi-edit context in an interpretable
descriptor framework. Use it to construct site-aware features, compare molecular
representations on fixed PAMPA partitions, and examine how the conclusions change
under sequence, temporal, and peptide-family shifts.

The accompanying paper, **Site-Conditioned Edit Chemistry for Cyclic Peptide
Permeability Modeling**, is accepted in *Journal of Cheminformatics*.
[Read the preprint](https://doi.org/10.21203/rs.3.rs-9541490/v1) ·
[Reproduction guide](docs/reproduction.md) · [Data and terms](docs/data.md) ·
[Citation](#citation)

## Method and archived results

The curated dataset contains **7,224 known-site PAMPA records**. Chemistry features
(A) describe molecular properties and edit counts; site features (B) describe
anchor positions and residues; context features (C) describe edit types,
attachments, and their combinations. Fixed splits, per-seed predictions,
hyperparameter trials, and uncertainty summaries accompany the code.

The archived 30-seed primary XGBoost benchmark gives the following mean test
R² values:

| Representation | Random split | Strict sequence-component split |
|---|---:|---:|
| Chemistry (A) | 0.427 | 0.074 |
| Chemistry + Site (A+B) | 0.444 | 0.131 |
| Chemistry + Site + Context (A+B+C) | 0.453 | 0.143 |

These results show a modest site-aware improvement within the random-split
setting and substantially lower performance under sequence shift. The strict
component split keeps related sequences together and has a small held-out set
(97 test records). Values above summarize the current repository's 30-seed
archive; the linked preprint v1 describes an earlier analysis version. See
[the reproduction guide](docs/reproduction.md#reference-results) for intervals,
split definitions, and the exact archived source revision.

## Quick start

Python 3.12 is the recorded research environment. Run these commands from the
repository root:

```bash
git clone https://github.com/leochenminrui/CycSite.git
cd CycSite
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Construct the full 73-feature representation for the first supplied peptide:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from data.pem_schema import PEMSample
from baselines.featurizers.anchor_aware_descriptors import AnchorAwareDescriptorFeaturizer

with Path("data/processed/pem_schema/cycpeptmpdb_pampa.jsonl").open() as stream:
    sample = PEMSample.model_validate_json(next(stream))
featurizer = AnchorAwareDescriptorFeaturizer(ablation_mode="full")
features = featurizer.featurize([sample])
print(sample.sample_id, features.shape)
PY
```

Expected output: `CYCPEPTMPDB_PAMPA_000000 (1, 73)`.
The input contains the peptide sequence, molecular SMILES, and resolved edit
anchors. The output is a descriptor matrix; model fitting and evaluation commands
are in the [reproduction guide](docs/reproduction.md).

## Reproducibility resources

Check the archived dataset, split, and summary checksums without installing the
scientific dependencies:

```bash
python3 scripts/check_release.py
```

The [reproduction guide](docs/reproduction.md) covers saved-result checks, dataset
audits, model training, sensitivity analyses, and figure generation. CPU-based
training is supported; the default primary benchmark runs up to eight workers,
with two XGBoost threads per fit. Full tuning comprises thousands of fits and is
separate from the quick start. The guide also records a
[version difference in the compact scaffold summary](docs/reproduction.md#compact-summary-consistency).

| Path | Contents |
|---|---|
| `src/` | Dataset schema, descriptor construction, tuning, and evaluation |
| `scripts/` | Executable data, benchmark, statistics, and figure workflows |
| `configs/benchmark/benchmark.yaml` | Primary benchmark and search settings |
| `data/` | Source CSVs, normalized records, and fixed partitions |
| `results/` | Saved predictions, trials, estimates, uncertainty, and manifests |
| `docs/` | Reproduction instructions and data provenance |

## Citation

If you use CycSite's method, code, or results, please cite the accompanying paper:

> Chen, M., Zou, G., Chen, M., Song, B., and Saigo, H. (2026).
> *Site-Conditioned Edit Chemistry for Cyclic Peptide Permeability Modeling*.
> Research Square, preprint v1. https://doi.org/10.21203/rs.3.rs-9541490/v1.
> Accepted in *Journal of Cheminformatics*.

The DOI currently provided here identifies the preprint. Cite the journal version
when its publication metadata become available. For computational reproducibility,
also report the repository commit used (`git rev-parse HEAD`).

```bibtex
@misc{chen2026cycsite,
  author = {Chen, Minrui and Zou, Guanjie and Chen, Minqi and Song, Bowen and Saigo, Hiroto},
  title = {Site-Conditioned Edit Chemistry for Cyclic Peptide Permeability Modeling},
  year = {2026},
  publisher = {Research Square},
  doi = {10.21203/rs.3.rs-9541490/v1},
  url = {https://doi.org/10.21203/rs.3.rs-9541490/v1},
  note = {Preprint v1. Accepted in Journal of Cheminformatics}
}
```

[CITATION.cff](CITATION.cff) provides machine-readable paper metadata for GitHub's
**Cite this repository** menu. If you use the supplied CycPeptMPDB records, also
cite the [source database paper](https://doi.org/10.1021/acs.jcim.2c01573).

## License

The code is released under the [MIT License](LICENSE). CycPeptMPDB-derived records
retain their source terms; the code license does not relicense third-party data.
See [data provenance and reuse](docs/data.md).
