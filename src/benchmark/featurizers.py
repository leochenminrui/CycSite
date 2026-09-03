"""
Featurizers used by the final CASPer benchmark.

Each featurizer is a callable class with a consistent API:
    featurizer.fit(train_samples)   # optional, for imputation / normalization
    X = featurizer.transform(samples)  # returns np.ndarray (n_samples, n_features)
    names = featurizer.get_feature_names()  # returns List[str]
"""

import numpy as np
from typing import List, Optional, Dict, Any

import sys
from pathlib import Path as _Path
_src_dir = _Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from data.pem_schema import PEMSample


# ─── Featurizer Registry ────────────────────────────────────────────────────

FEATURIZER_REGISTRY: Dict[str, type] = {}


def register_featurizer(name: str):
    """Decorator to register a featurizer class."""
    def wrapper(cls):
        FEATURIZER_REGISTRY[name] = cls
        return cls
    return wrapper


# ─── Helper: AA properties ──────────────────────────────────────────────────

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

HYDROPHOBIC_AA = set("ILVAFWM")
CHARGED_AA = set("DEKRH")
POLAR_AA = set("STNQYC")
AROMATIC_AA = set("FYW")
SMALL_AA = set("GAS")

AA_MW = {
    'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
    'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
    'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
    'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181,
}

AA_HYDRO = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

AA_PROPERTY_GROUPS = {
    'hydrophobic': 'AILMFVPW',
    'polar': 'NQST',
    'positive': 'KRH',
    'negative': 'DE',
    'aromatic': 'FYW',
    'small': 'AGSV',
    'tiny': 'AGS',
    'aliphatic': 'ILV',
}


# ─── A. Internal Baseline Featurizers ───────────────────────────────────────

@register_featurizer("aa_composition")
class AACompositionFeaturizer:
    """AA composition + basic sequence properties (33 dim)."""
    def __init__(self, use_aa_composition=True, use_property_composition=True,
                 use_basic_features=True, use_dipeptide=False):
        self.use_aa_composition = use_aa_composition
        self.use_property_composition = use_property_composition
        self.use_basic_features = use_basic_features
        self.use_dipeptide = use_dipeptide
        self._dim = 0
        if use_aa_composition: self._dim += 20
        if use_property_composition: self._dim += len(AA_PROPERTY_GROUPS)
        if use_basic_features: self._dim += 5
        if use_dipeptide: self._dim += 400

    def fit(self, samples): pass

    def transform(self, samples):
        X = np.zeros((len(samples), self._dim))
        for i, s in enumerate(samples):
            feats = []
            seq = s.sequence.upper()
            L = max(len(seq), 1)
            if self.use_aa_composition:
                feats.extend([seq.count(aa) / L for aa in STANDARD_AA])
            if self.use_property_composition:
                feats.extend([sum(seq.count(aa) for aa in grp) / L
                             for grp in AA_PROPERTY_GROUPS.values()])
            if self.use_basic_features:
                mw = sum(AA_MW.get(aa, 0) for aa in seq)
                charge = seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')
                hydro = sum(AA_HYDRO.get(aa, 0) for aa in seq) / L
                aroma = (seq.count('F') + seq.count('Y') + seq.count('W')) / L
                feats.extend([float(L), mw, float(charge), hydro, aroma])
            if self.use_dipeptide:
                di = np.zeros(400)
                for j in range(len(seq) - 1):
                    idx = STANDARD_AA.find(seq[j]) * 20 + STANDARD_AA.find(seq[j + 1])
                    if idx >= 0:
                        di[idx] += 1
                di /= max(len(seq) - 1, 1)
                feats.extend(di.tolist())
            X[i] = feats
        return X.astype(np.float32)

    def get_feature_names(self):
        names = []
        if self.use_aa_composition:
            names.extend([f'aa_{aa}' for aa in STANDARD_AA])
        if self.use_property_composition:
            names.extend([f'prop_{k}' for k in AA_PROPERTY_GROUPS])
        if self.use_basic_features:
            names.extend(['length', 'mw', 'charge', 'hydro', 'aromatic'])
        if self.use_dipeptide:
            for a1 in STANDARD_AA:
                for a2 in STANDARD_AA:
                    names.append(f'di_{a1}{a2}')
        return names


# ─── B. Anchor-Aware Group Featurizers (reuse existing featurizer) ──────────

@register_featurizer("anchor_aware")
class AnchorAwareWrapper:
    """Wrapper around existing AnchorAwareDescriptorFeaturizer."""
    def __init__(self, descriptor_set="basic", ablation_mode="full"):
        from baselines.featurizers.anchor_aware_descriptors import \
            AnchorAwareDescriptorFeaturizer
        self._inner = AnchorAwareDescriptorFeaturizer(
            descriptor_set=descriptor_set, ablation_mode=ablation_mode)

    def fit(self, samples): pass

    def transform(self, samples):
        return self._inner.featurize(samples)

    def get_feature_names(self):
        return self._inner.get_feature_names()


@register_featurizer("site_only")
class SiteOnlyFeaturizer:
    """Group B features only: B1 + B2 + B3 (35 dim), no chemistry."""
    def __init__(self):
        from baselines.featurizers.anchor_aware_descriptors import \
            AnchorAwareDescriptorFeaturizer
        self._full = AnchorAwareDescriptorFeaturizer(
            descriptor_set="basic", ablation_mode="full")

    def fit(self, samples): pass

    def transform(self, samples):
        X = np.zeros((len(samples), 35))
        for i, s in enumerate(samples):
            # Trigger internal extraction to populate _last_b1/b2/b3
            self._full.featurize_sample(s)
            feats = list(self._full._last_b1) + list(self._full._last_b2) + list(self._full._last_b3)
            X[i] = feats
        return X.astype(np.float32)

    def get_feature_names(self):
        all_names = self._full.get_feature_names()
        # Names after the 10 chemistry features:
        # B1=6, B2=20, B3=9 → indices 10:45
        return all_names[10:45]


@register_featurizer("context_only")
class ContextOnlyFeaturizer:
    """Group C features only (28 dim), no chemistry or site."""
    def __init__(self):
        from baselines.featurizers.anchor_aware_descriptors import \
            AnchorAwareDescriptorFeaturizer
        self._full = AnchorAwareDescriptorFeaturizer(
            descriptor_set="basic", ablation_mode="full")

    def fit(self, samples): pass

    def transform(self, samples):
        X = np.zeros((len(samples), 35))  # C features are 28 dim but we'll compute
        for i, s in enumerate(samples):
            c_feats = self._full._extract_attachment_aware_features(s)
            X[i, :len(c_feats)] = c_feats
        return X[:, :len(c_feats)].astype(np.float32)

    def get_feature_names(self):
        all_names = self._full.get_feature_names()
        return all_names[45:]  # C features start after A(10) + B(35)


# ═══ C. Generic Chemistry Featurizers ═══════════════════════════════════════

def _get_sample_smiles(sample: PEMSample) -> Optional[str]:
    """Get SMILES from assay_metadata."""
    return (sample.assay_metadata or {}).get('smiles')


@register_featurizer("ecfp")
class ECFPFeaturizer:
    """Fixed-length Morgan fingerprint (ECFP) from peptide SMILES."""
    def __init__(self, radius=2, nBits=2048, aggregation="mean"):
        from rdkit.Chem import rdFingerprintGenerator
        self.radius = radius
        self.nBits = nBits
        self.aggregation = aggregation
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.nBits
        )

    def fit(self, samples): pass

    def _compute_fp(self, smiles: str) -> Optional[np.ndarray]:
        from rdkit import Chem, DataStructs
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = self._generator.GetFingerprint(mol)
            arr = np.zeros(self.nBits)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return None

    def transform(self, samples):
        X = np.zeros((len(samples), self.nBits), dtype=np.float32)
        for i, s in enumerate(samples):
            smiles = _get_sample_smiles(s)
            if smiles:
                fp = self._compute_fp(smiles)
                if fp is not None:
                    X[i] = fp
            # If no SMILES / invalid, leave as zeros
        return X

    def get_feature_names(self):
        return [f'ecfp_{j}' for j in range(self.nBits)]


@register_featurizer("rdkit_full")
class RDKitFullFeaturizer:
    """Official RDKit 2D descriptor registry with train-only imputation."""
    def __init__(self, aggregation="mean"):
        self.aggregation = aggregation
        self._desc_names = None
        self._nan_imputation_vals = None  # median per feature, fit on train

    def _get_all_descriptor_names(self):
        from rdkit.Chem import Descriptors
        # Descriptors.descList is RDKit's public, versioned registry of
        # molecule-to-scalar 2D descriptors.  Using it avoids accidentally
        # collecting helper/meta functions through module introspection.
        return [name for name, _ in Descriptors.descList]

    def _compute_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            vals = []
            for name in self._desc_names:
                try:
                    v = getattr(Descriptors, name)(mol)
                    vals.append(float(v) if v is not None else np.nan)
                except Exception:
                    vals.append(np.nan)
            return np.array(vals, dtype=np.float64)
        except Exception:
            return None

    def fit(self, samples):
        self._desc_names = self._get_all_descriptor_names()
        # Compute descriptors for all samples, collect medians for imputation
        all_descs = []
        for s in samples:
            smiles = _get_sample_smiles(s)
            if smiles:
                d = self._compute_descriptors(smiles)
                if d is not None:
                    all_descs.append(d)
        if all_descs:
            arr = np.array(all_descs)
            self._nan_imputation_vals = np.nanmedian(arr, axis=0)
            self._nan_imputation_vals = np.nan_to_num(
                self._nan_imputation_vals, nan=0.0)
        else:
            self._nan_imputation_vals = np.zeros(len(self._desc_names))

    def transform(self, samples):
        if self._desc_names is None:
            self.fit(samples)
        n_feat = len(self._desc_names)
        X = np.zeros((len(samples), n_feat), dtype=np.float32)
        imp = self._nan_imputation_vals if self._nan_imputation_vals is not None \
            else np.zeros(n_feat)
        for i, s in enumerate(samples):
            smiles = _get_sample_smiles(s)
            if smiles:
                d = self._compute_descriptors(smiles)
                if d is not None:
                    # Impute NaN with median
                    d = np.where(np.isnan(d), imp, d)
                    X[i] = d
                else:
                    X[i] = imp
            else:
                X[i] = imp
        return X

    def get_feature_names(self):
        if self._desc_names is None:
            self._desc_names = self._get_all_descriptor_names()
        return [f'rdkit_{n}' for n in self._desc_names]
