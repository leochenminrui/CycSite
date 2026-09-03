"""
PEM Unified Sample Schema Implementation.

Pydantic models for strict validation of the PEM schema specification.
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime
import re


# Controlled vocabularies as Enums

class EditFamily(str, Enum):
    """Edit family categorization."""
    N_TERMINAL = "n_terminal"
    C_TERMINAL = "c_terminal"
    SIDECHAIN = "sidechain"
    BACKBONE = "backbone"
    CYCLIZATION = "cyclization"
    SUBSTITUTION = "substitution"
    OTHER = "other"


class AnchorKind(str, Enum):
    """Anchor position classification."""
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    GLOBAL = "global"


class ParserStatus(str, Enum):
    """Parsing confidence level."""
    SUCCESS = "success"
    INFERRED = "inferred"
    PARTIAL = "partial"
    FAILED = "failed"


class AnchorStatus(str, Enum):
    """Sample-level anchor resolvability."""
    EXPLICIT_ANCHOR = "explicit_anchor"
    WEAKLY_INFERABLE = "weakly_inferable"
    NOT_RESOLVABLE = "not_resolvable"
    NO_EDITS = "no_edits"


# Standard amino acid alphabet
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


class Edit(BaseModel):
    """
    Chemical modification/edit to peptide backbone.

    Represents a single chemical modification with full anchor
    and parsing information.
    """

    # Identity
    edit_id: str = Field(
        ...,
        description="Unique identifier for this edit",
        pattern=r"^[A-Z_]+_[0-9]+_edit_[0-9]+$"
    )

    # Classification
    edit_family: EditFamily = Field(
        ...,
        description="High-level edit category"
    )

    edit_type: str = Field(
        ...,
        min_length=1,
        description="Specific modification type"
    )

    # Anchor information
    anchor_kind: AnchorKind = Field(
        ...,
        description="Anchor classification (explicit/inferred/ambiguous/global)"
    )

    anchor_positions: List[int] = Field(
        ...,
        description="0-indexed positions in sequence (empty for global mods)"
    )

    anchor_residues: List[str] = Field(
        ...,
        description="Amino acids at anchor positions (empty for terminal/global)"
    )

    # Chemical representation
    chem_rep_raw: str = Field(
        ...,
        min_length=1,
        description="Original notation from source data"
    )

    chem_rep_canonical: str = Field(
        ...,
        min_length=1,
        description="Standardized representation (HELM/SMILES/custom)"
    )

    # Attachment
    attachment_semantics: str = Field(
        ...,
        min_length=1,
        description="Where/how modification attaches"
    )

    # Parsing metadata
    parser_status: ParserStatus = Field(
        ...,
        description="Parsing confidence level"
    )

    rule_id: str = Field(
        ...,
        min_length=1,
        description="Parsing rule identifier"
    )

    edit_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional edit-specific information"
    )

    @field_validator('anchor_positions')
    @classmethod
    def positions_must_be_sorted(cls, v):
        """Anchor positions must be sorted and unique."""
        if v != sorted(set(v)):
            raise ValueError("Anchor positions must be sorted and unique")
        return v

    @field_validator('anchor_residues')
    @classmethod
    def residues_must_be_valid(cls, v):
        """Anchor residues must be valid amino acids."""
        for residue in v:
            if residue and residue not in STANDARD_AA:
                raise ValueError(f"Invalid amino acid: {residue}")
        return v

    @model_validator(mode='after')
    def validate_anchor_consistency(self):
        """Validate consistency between anchor fields."""
        positions = self.anchor_positions
        residues = self.anchor_residues

        # Positions and residues must have same length
        if len(positions) != len(residues):
            raise ValueError(
                f"Mismatch: {len(positions)} positions but {len(residues)} residues"
            )

        # Global anchors should have empty positions/residues
        if self.anchor_kind == AnchorKind.GLOBAL:
            if positions or residues:
                raise ValueError("Global anchor should have empty positions/residues")

        return self

    class Config:
        use_enum_values = True  # Serialize enums as strings


class PEMSample(BaseModel):
    """
    Unified PEM sample schema.

    Represents a peptide sample with complete experimental data,
    chemical modifications, and provenance.
    """

    # Core identity
    sample_id: str = Field(
        ...,
        description="Unique sample identifier",
        pattern=r"^[A-Z_]+_[0-9]+$"
    )

    dataset: str = Field(
        ...,
        min_length=1,
        description="Source dataset name"
    )

    # Sequence (backbone only)
    sequence: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Canonical amino acid sequence (backbone)"
    )

    # Experimental measurement
    label: float = Field(
        ...,
        description="Primary experimental measurement"
    )

    label_type: str = Field(
        ...,
        min_length=1,
        description="Type of measurement (e.g., log_permeability)"
    )

    label_unit: str = Field(
        ...,
        min_length=1,
        description="Unit of measurement"
    )

    # Assay information
    assay_type: str = Field(
        ...,
        min_length=1,
        description="Specific assay (e.g., PAMPA)"
    )

    assay_metadata: Dict[str, Any] = Field(
        ...,
        min_items=1,
        description="Full experimental conditions"
    )

    # Chemical modifications
    edits: List[Edit] = Field(
        default_factory=list,
        description="Chemical modifications to backbone"
    )

    # Anchor classification
    anchor_status: AnchorStatus = Field(
        ...,
        description="Overall anchor resolvability"
    )

    # Provenance
    provenance: Dict[str, Any] = Field(
        ...,
        description="Source file and parsing information"
    )

    # Optional split assignment
    split_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Train/val/test assignment (optional)"
    )

    # Quality flags
    quality_flags: List[str] = Field(
        default_factory=list,
        description="Quality concerns or warnings"
    )

    @field_validator('sequence')
    @classmethod
    def sequence_must_be_valid(cls, v):
        """Sequence must contain only standard amino acids."""
        v_upper = v.upper()
        invalid = set(v_upper) - STANDARD_AA
        if invalid:
            raise ValueError(f"Invalid amino acids in sequence: {invalid}")
        return v_upper

    @field_validator('label')
    @classmethod
    def label_must_be_finite(cls, v):
        """Label must be finite (no NaN or Inf)."""
        import math
        if not math.isfinite(v):
            raise ValueError(f"Label must be finite, got {v}")
        return v

    @field_validator('provenance')
    @classmethod
    def provenance_must_have_required_fields(cls, v):
        """Provenance must contain required fields."""
        required = ['source_file', 'source_row_index', 'parser_version']
        missing = [field for field in required if field not in v]
        if missing:
            raise ValueError(f"Provenance missing required fields: {missing}")
        return v

    @model_validator(mode='after')
    def validate_edit_positions(self):
        """Validate that all edit positions are valid for sequence."""
        sequence = self.sequence
        edits = self.edits
        seq_len = len(sequence)

        for edit in edits:
            for pos in edit.anchor_positions:
                if pos < 0 or pos >= seq_len:
                    raise ValueError(
                        f"Edit {edit.edit_id} has invalid position {pos} "
                        f"for sequence length {seq_len}"
                    )

        return self

    @model_validator(mode='after')
    def validate_edit_residues(self):
        """Validate that edit residues match sequence."""
        sequence = self.sequence
        edits = self.edits

        for edit in edits:
            for pos, residue in zip(edit.anchor_positions, edit.anchor_residues):
                if residue and sequence[pos] != residue:
                    raise ValueError(
                        f"Edit {edit.edit_id} claims residue {residue} at position {pos}, "
                        f"but sequence has {sequence[pos]}"
                    )

        return self

    @model_validator(mode='after')
    def validate_anchor_status_consistency(self):
        """Validate anchor_status matches edit classifications."""
        edits = self.edits
        anchor_status = self.anchor_status

        if not edits:
            if anchor_status != AnchorStatus.NO_EDITS:
                raise ValueError(
                    f"No edits but anchor_status is {anchor_status}, "
                    "should be 'no_edits'"
                )
        else:
            # If all edits are explicit, overall should be explicit_anchor
            anchor_kinds = {edit.anchor_kind for edit in edits}

            if anchor_kinds == {AnchorKind.EXPLICIT}:
                if anchor_status != AnchorStatus.EXPLICIT_ANCHOR:
                    raise ValueError(
                        "All edits have explicit anchors but sample anchor_status "
                        f"is {anchor_status}"
                    )
            elif AnchorKind.INFERRED in anchor_kinds or AnchorKind.AMBIGUOUS in anchor_kinds:
                if anchor_status not in [
                    AnchorStatus.WEAKLY_INFERABLE,
                    AnchorStatus.NOT_RESOLVABLE
                ]:
                    raise ValueError(
                        "Edits have inferred/ambiguous anchors but sample anchor_status "
                        f"is {anchor_status}"
                    )

        return self

    class Config:
        use_enum_values = True


class SchemaValidationReport(BaseModel):
    """
    Report of schema validation results.

    Summarizes validation success/failure across a dataset.
    """
    dataset: str
    total_samples: int
    valid_samples: int
    invalid_samples: int
    validation_errors: Dict[str, List[str]]  # error_type -> [sample_ids]
    timestamp: str

    @classmethod
    def create(
        cls,
        dataset: str,
        samples: List[PEMSample],
        errors: Dict[str, List[str]]
    ):
        """Create validation report from samples and errors."""
        return cls(
            dataset=dataset,
            total_samples=len(samples) + sum(len(v) for v in errors.values()),
            valid_samples=len(samples),
            invalid_samples=sum(len(v) for v in errors.values()),
            validation_errors=errors,
            timestamp=datetime.now().isoformat()
        )


# Helper functions for serialization

def sample_to_dict(sample: PEMSample) -> Dict[str, Any]:
    """Convert PEMSample to dictionary for serialization."""
    return sample.dict()


def sample_from_dict(data: Dict[str, Any]) -> PEMSample:
    """Create PEMSample from dictionary."""
    return PEMSample(**data)


def validate_sample(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate a sample dictionary.

    Args:
        data: Sample data dictionary

    Returns:
        (is_valid, error_message)
    """
    try:
        PEMSample(**data)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_samples_batch(
    samples: List[Dict[str, Any]]
) -> tuple[List[PEMSample], Dict[str, List[str]]]:
    """
    Validate a batch of samples.

    Args:
        samples: List of sample data dictionaries

    Returns:
        (valid_samples, errors_by_type)
    """
    valid_samples = []
    errors = {}

    for i, sample_data in enumerate(samples):
        try:
            sample = PEMSample(**sample_data)
            valid_samples.append(sample)
        except Exception as e:
            error_type = type(e).__name__
            if error_type not in errors:
                errors[error_type] = []
            sample_id = sample_data.get('sample_id', f'UNKNOWN_{i}')
            errors[error_type].append(f"{sample_id}: {str(e)}")

    return valid_samples, errors
