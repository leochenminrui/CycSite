# Data provenance and reuse

## Source and fixed snapshot

The source records come from [CycPeptMPDB](http://cycpeptmpdb.com/), a database
assembled from published papers and patents. Its
[download page](http://cycpeptmpdb.com/download/) provides the upstream tables;
the database may change after the snapshot used here.

| Supplied file | Records | SHA-256 |
|---|---:|---|
| `data/raw/CycPeptMPDB_Peptide_All.csv` | 8,466 | `54e2be28199c0c3a8d6d0923192161d418eca8efd5c9b804e9bceae2faa52e51` |
| `data/raw/CycPeptMPDB_Peptide_Assay_PAMPA.csv` | 7,298 | `02da1cfc18a92b3ae6e70152445b23c05ce6bb0b6ed10fc7c9e141fbd9462fde` |

The supplied normalized PAMPA JSONL retains 7,224 records with resolved site
annotations. The fixed partition files and their hashes are listed in
[`results/RESULTS_MANIFEST.json`](../results/RESULTS_MANIFEST.json). Use the
bundled snapshot and fixed partitions for the archived analysis rather than
silently replacing them with a newer database download.

Each normalized record includes its source row, original database ID, source
publication, parser version, molecular representation, and edit annotations.
The PAMPA measurement is used as a proxy for passive membrane permeability.

## Reuse terms and attribution

The repository's MIT license applies to its software. It does not grant new
rights to CycPeptMPDB records or their underlying third-party contributions.
Follow the source database's current terms and the attribution and permissions
applicable to the originating publications or patents when reusing or
redistributing data. No independent data license is assigned by this repository.

When using these records, cite the source database in addition to the CycSite
paper:

> Li, J., Yanagisawa, K., Sugita, M., Fujie, T., Ohue, M., and Akiyama, Y. (2023).
> CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic
> Peptides. *Journal of Chemical Information and Modeling*, 63(7), 2240–2250.
> https://doi.org/10.1021/acs.jcim.2c01573.

The [database paper](https://doi.org/10.1021/acs.jcim.2c01573) describes the
collection and links to the originating sources. The article's open-access
license should not be assumed to license every database asset separately.
