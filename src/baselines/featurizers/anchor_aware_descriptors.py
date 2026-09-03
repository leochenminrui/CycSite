"""
Anchor-Aware Descriptor Featurizer.

Implements the anchor-aware descriptor feature schema for XGBoost baseline.
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional, Set
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from collections import Counter
import warnings

from data.pem_schema import PEMSample


# Amino acid property groups
HYDROPHOBIC_AA = set("ILVAFWM")
CHARGED_AA = set("DEKRH")
POLAR_AA = set("STNQYC")
AROMATIC_AA = set("FYW")
SMALL_AA = set("GAS")


# Canonical feature-to-block schema for the frozen basic descriptor system.
# Keep this list aligned with feature construction below; the SHAP workflow
# serializes it to a machine-readable mapping for manuscript consistency.
B1_FEATURE_NAMES = (
    'anchor_count_total',
    'anchor_count_unique',
    'anchor_density',
    'anchor_pos_mean',
    'anchor_pos_std',
    'anchor_pos_range',
)
B2_FEATURE_NAMES = tuple(f'anchor_res_{aa}' for aa in "ACDEFGHIKLMNPQRSTVWY")
B3_FEATURE_NAMES = (
    'anchor_hydrophobic_frac',
    'anchor_charged_frac',
    'anchor_polar_frac',
    'anchor_aromatic_frac',
    'anchor_small_frac',
    'anchor_proline_frac',
    'anchor_terminal_frac',
    'anchor_n_terminal_count',
    'anchor_c_terminal_count',
)
C_FEATURE_NAMES = (
    'edit_family_n_terminal',
    'edit_family_c_terminal',
    'edit_family_sidechain',
    'edit_family_backbone',
    'edit_family_cyclization',
    'edit_family_substitution',
    'edit_family_other',
    'edit_type_count_unique',
    'edit_type_entropy',
    'attachment_type_count_unique',
    'has_cyclization',
    'has_backbone_mod',
    'edit_n_methylation',
    'edit_d_amino_acid',
    'edit_non_standard_aa',
    'edit_head_to_tail_cyclization',
    'edit_sidechain_cyclization',
    'edit_disulfide',
    'edit_acetylation',
    'edit_amidation',
    'edit_phosphorylation',
    'edit_other_types',
    'anchor_pair_count',
    'cyclization_ring_size',
    'edit_anchor_overlap',
    'sequence_length',
    'sequence_cyclic',
    'modification_rate',
)


def get_feature_block_mapping() -> Dict[str, str]:
    """Return the canonical feature-to-block map for the frozen basic schema."""
    a_names = (
        'mol_weight', 'logp', 'tpsa', 'num_h_acceptors', 'num_h_donors',
        'num_rotatable_bonds', 'num_aromatic_rings', 'num_aliphatic_rings',
        'num_edits', 'num_edit_families',
    )
    mapping = {name: 'A' for name in a_names}
    mapping.update({name: 'B1' for name in B1_FEATURE_NAMES})
    mapping.update({name: 'B2' for name in B2_FEATURE_NAMES})
    mapping.update({name: 'B3' for name in B3_FEATURE_NAMES})
    mapping.update({name: 'C' for name in C_FEATURE_NAMES})
    return mapping


class AnchorAwareDescriptorFeaturizer:
    """
    Anchor-aware descriptor featurizer.

    Combines:
    - Global chemistry features (whole-molecule descriptors)
    - Anchor-aware local features (position, residue, context)
    - Attachment-aware multi-edit features (edit types, structure)
    """

    def __init__(self, descriptor_set: str = "basic", ablation_mode: str = "full"):
        """
        Initialize anchor-aware descriptor featurizer.

        Args:
            descriptor_set: The final study uses the basic descriptor set.
            ablation_mode: Feature groups to include:
                - "full": All features (A + B + C)
                - "chemistry_only": Global chemistry only (A)
                - "chemistry_attachment": A + C (attachment features)
                - "chemistry_anchors": A + B (all anchor features)
                - "site_context_only": B + C
        """
        self.descriptor_set = descriptor_set
        self.ablation_mode = ablation_mode

        # Define descriptor functions for global chemistry
        if descriptor_set == "basic":
            self.descriptor_funcs = {
                'mol_weight': Descriptors.MolWt,
                'logp': Descriptors.MolLogP,
                'tpsa': Descriptors.TPSA,
                'num_h_acceptors': Descriptors.NumHAcceptors,
                'num_h_donors': Descriptors.NumHDonors,
                'num_rotatable_bonds': Descriptors.NumRotatableBonds,
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings,
                'num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings,
            }
        else:
            raise ValueError(f"Unknown descriptor set: {descriptor_set}")

    def _extract_global_chemistry_features(self, sample: PEMSample) -> List[float]:
        """Extract global chemistry features from SMILES."""
        features = []

        # Get SMILES
        smiles = sample.assay_metadata.get('smiles')

        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Compute descriptors
                    for func in self.descriptor_funcs.values():
                        try:
                            value = func(mol)
                            features.append(float(value) if value is not None else 0.0)
                        except Exception:
                            features.append(0.0)
                else:
                    # Invalid SMILES
                    features.extend([0.0] * len(self.descriptor_funcs))
            except Exception:
                # Parsing error
                features.extend([0.0] * len(self.descriptor_funcs))
        else:
            # No SMILES
            features.extend([0.0] * len(self.descriptor_funcs))

        # Add edit counts
        features.append(float(len(sample.edits)))
        features.append(float(len(set(e.edit_family for e in sample.edits))))

        return features

    def _extract_anchor_positions_and_residues(self, sample: PEMSample) -> tuple:
        """Extract all anchor positions and residues from edits."""
        all_positions = []
        all_residues = []

        for edit in sample.edits:
            # Skip global edits
            if edit.anchor_kind == "global" or not edit.anchor_positions:
                continue

            # Add positions and residues
            all_positions.extend(edit.anchor_positions)
            all_residues.extend(edit.anchor_residues)

        return all_positions, all_residues

    def _extract_anchor_aware_local_features(self, sample: PEMSample) -> List[float]:
        """Extract anchor-aware local features."""
        features = []

        # Get anchor positions and residues
        all_positions, all_residues = self._extract_anchor_positions_and_residues(sample)

        # B1. Anchor Position Statistics (6 features)
        b1_features = []
        if all_positions:
            anchor_count_total = len(all_positions)
            anchor_count_unique = len(set(all_positions))
            sequence_length = len(sample.sequence)
            if sequence_length > 1:
                normalized_positions = [
                    float(position) / float(sequence_length - 1)
                    for position in all_positions
                ]
            else:
                normalized_positions = [0.0 for _ in all_positions]

            b1_features.append(float(anchor_count_total))
            b1_features.append(float(anchor_count_unique))
            b1_features.append(float(anchor_count_unique) / max(sequence_length, 1))
            b1_features.append(float(np.mean(normalized_positions)))
            b1_features.append(float(np.std(normalized_positions)) if len(normalized_positions) > 1 else 0.0)
            b1_features.append(float(max(normalized_positions) - min(normalized_positions)) if len(normalized_positions) > 1 else 0.0)
        else:
            b1_features.extend([0.0] * 6)

        # B2. Anchor Residue Composition (20 features)
        # One-hot encoding for each standard amino acid
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        b2_features = []
        if all_residues:
            residue_counts = Counter(all_residues)
            total_residues = len(all_residues)
            for aa in standard_aa:
                b2_features.append(float(residue_counts.get(aa, 0)) / total_residues)
        else:
            b2_features.extend([0.0] * 20)

        # B3. Anchor Local Context (9 features)
        b3_features = []
        if all_residues:
            total = len(all_residues)
            residue_set = set(all_residues)

            hydrophobic_frac = sum(1 for r in all_residues if r in HYDROPHOBIC_AA) / total
            charged_frac = sum(1 for r in all_residues if r in CHARGED_AA) / total
            polar_frac = sum(1 for r in all_residues if r in POLAR_AA) / total
            aromatic_frac = sum(1 for r in all_residues if r in AROMATIC_AA) / total
            small_frac = sum(1 for r in all_residues if r in SMALL_AA) / total
            proline_frac = sum(1 for r in all_residues if r == 'P') / total

            terminal_count = sum(1 for e in sample.edits if e.edit_family in ["n_terminal", "c_terminal"])
            n_terminal_count = sum(1 for e in sample.edits if e.edit_family == "n_terminal")
            c_terminal_count = sum(1 for e in sample.edits if e.edit_family == "c_terminal")

            b3_features.extend([
                hydrophobic_frac,
                charged_frac,
                polar_frac,
                aromatic_frac,
                small_frac,
                proline_frac,
                float(terminal_count) / max(len(sample.edits), 1),
                float(n_terminal_count),
                float(c_terminal_count),
            ])
        else:
            b3_features.extend([0.0] * 9)

        # Store components for ablation
        self._last_b1 = b1_features
        self._last_b2 = b2_features
        self._last_b3 = b3_features

        # Combine all B features
        features.extend(b1_features)
        features.extend(b2_features)
        features.extend(b3_features)

        return features

    def _extract_attachment_aware_features(self, sample: PEMSample) -> List[float]:
        """Extract attachment-aware multi-edit features."""
        features = []

        # C1. Edit Family Distribution (7 features)
        edit_families = [e.edit_family for e in sample.edits]
        family_counts = Counter(edit_families)

        for family in ["n_terminal", "c_terminal", "sidechain", "backbone", "cyclization", "substitution", "other"]:
            features.append(float(family_counts.get(family, 0)))

        # C2. Edit Type Diversity (5 features)
        edit_types = [e.edit_type for e in sample.edits]
        attachment_types = [e.attachment_semantics for e in sample.edits]

        edit_type_count_unique = len(set(edit_types)) if edit_types else 0
        features.append(float(edit_type_count_unique))

        # Shannon entropy for edit types
        if edit_types:
            type_counts = Counter(edit_types)
            total = len(edit_types)
            entropy = -sum((count / total) * np.log2(count / total) for count in type_counts.values())
            features.append(entropy)
        else:
            features.append(0.0)

        attachment_type_count_unique = len(set(attachment_types)) if attachment_types else 0
        features.append(float(attachment_type_count_unique))

        has_cyclization = float(any(e.edit_family == "cyclization" for e in sample.edits))
        has_backbone_mod = float(any(e.edit_family == "backbone" for e in sample.edits))
        features.extend([has_cyclization, has_backbone_mod])

        # C3. Specific Edit Type Counts (10 features)
        edit_type_counts = Counter(edit_types)

        specific_types = [
            "n_methylation",
            "d_amino_acid",
            "non_standard_aa",
            "head_to_tail_cyclization",
            "sidechain_cyclization",
            "disulfide",
            "acetylation",
            "amidation",
            "phosphorylation",
        ]

        for edit_type in specific_types:
            features.append(float(edit_type_counts.get(edit_type, 0)))

        # Other types
        other_count = sum(count for edit_type, count in edit_type_counts.items() if edit_type not in specific_types)
        features.append(float(other_count))

        # C4. Multi-Edit Aggregation (6 features)
        anchor_pair_count = sum(1 for e in sample.edits if len(e.anchor_positions) > 1)
        features.append(float(anchor_pair_count))

        # Extract cyclization ring size if available
        cyclization_ring_size = 0.0
        for e in sample.edits:
            if e.edit_family == "cyclization" and e.edit_metadata:
                ring_size = e.edit_metadata.get("ring_size")
                if ring_size:
                    cyclization_ring_size = float(ring_size)
                    break
        features.append(cyclization_ring_size)

        # Anchor overlap
        all_positions, _ = self._extract_anchor_positions_and_residues(sample)
        position_counts = Counter(all_positions)
        edit_anchor_overlap = sum(1 for count in position_counts.values() if count > 1)
        features.append(float(edit_anchor_overlap))

        # Sequence-level features
        sequence_length = len(sample.sequence)
        sequence_cyclic = float(any(e.edit_family == "cyclization" for e in sample.edits))
        modification_rate = len(sample.edits) / max(sequence_length, 1)

        features.extend([
            float(sequence_length),
            sequence_cyclic,
            modification_rate,
        ])

        return features

    def featurize_sample(self, sample: PEMSample) -> np.ndarray:
        """
        Featurize a single sample.

        Args:
            sample: PEMSample with edits and SMILES

        Returns:
            Feature vector (size depends on ablation_mode)
        """
        features = []

        # A. Global Chemistry Features (10 features)
        a_features = self._extract_global_chemistry_features(sample)
        features.extend(a_features)

        # B. Anchor-Aware Local Features (35 features)
        # Extract all B features to populate self._last_b1, b2, b3
        self._extract_anchor_aware_local_features(sample)

        # C. Attachment-Aware Multi-Edit Features (28 features)
        c_features = self._extract_attachment_aware_features(sample)

        # Apply ablation mode
        if self.ablation_mode == "full":
            # All features: A + B + C
            features.extend(self._last_b1)
            features.extend(self._last_b2)
            features.extend(self._last_b3)
            features.extend(c_features)
        elif self.ablation_mode == "chemistry_only":
            # Only A
            pass  # Already added
        elif self.ablation_mode == "chemistry_attachment":
            # A + C
            features.extend(c_features)
        elif self.ablation_mode == "chemistry_anchors":
            # A + B (all anchor features)
            features.extend(self._last_b1)
            features.extend(self._last_b2)
            features.extend(self._last_b3)
        elif self.ablation_mode == "site_context_only":
            # B + C only (no chemistry A), for BC model
            # Strip the A features already added at start
            features = []  # clear A features
            features.extend(self._last_b1)
            features.extend(self._last_b2)
            features.extend(self._last_b3)
            features.extend(c_features)
        else:
            raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")

        return np.array(features, dtype=np.float32)

    def featurize(self, samples: List[PEMSample]) -> np.ndarray:
        """
        Featurize multiple samples.

        Args:
            samples: List of PEMSample objects

        Returns:
            Feature matrix (n_samples, 73)
        """
        feature_vectors = []
        for sample in samples:
            feat = self.featurize_sample(sample)
            feature_vectors.append(feat)

        return np.array(feature_vectors)

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []

        # A. Global Chemistry Features (10)
        names.extend(list(self.descriptor_funcs.keys()))
        names.extend(['num_edits', 'num_edit_families'])

        # B. Anchor-Aware Local Features (35)
        # B1. Position stats (6)
        names.extend(B1_FEATURE_NAMES)

        # B2. Residue composition (20)
        names.extend(B2_FEATURE_NAMES)

        # B3. Local context (9)
        names.extend(B3_FEATURE_NAMES)

        # C. Attachment-Aware Multi-Edit Features (28)
        # C1. Edit family distribution (7)
        names.extend(C_FEATURE_NAMES)

        # Project the canonical A/B1/B2/B3/C name list using exactly the same
        # slices as featurize_sample.  Returning all 73 names for an ablated
        # matrix silently mislabels feature importances and SHAP values.
        a, b1, b2, b3, c = names[:10], names[10:16], names[16:36], names[36:45], names[45:73]
        mode_names = {
            'full': a + b1 + b2 + b3 + c,
            'chemistry_only': a,
            'chemistry_attachment': a + c,
            'chemistry_anchors': a + b1 + b2 + b3,
            'site_context_only': b1 + b2 + b3 + c,
        }
        if self.ablation_mode not in mode_names:
            raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")
        return mode_names[self.ablation_mode]
