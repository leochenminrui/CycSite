"""
Base converter class for dataset-to-PEM schema conversion.

Provides common functionality for all dataset converters.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..pem_schema import PEMSample, Edit, AnchorStatus, AnchorKind, ParserStatus, EditFamily


class BaseConverter(ABC):
    """
    Abstract base class for dataset converters.

    Subclasses must implement dataset-specific parsing logic.
    """

    def __init__(
        self,
        dataset_name: str,
        parser_version: str = "1.0.0",
        strict_mode: bool = True
    ):
        """
        Initialize converter.

        Args:
            dataset_name: Name of source dataset
            parser_version: Version of parser logic
            strict_mode: If True, fail on ambiguous cases
        """
        self.dataset_name = dataset_name
        self.parser_version = parser_version
        self.strict_mode = strict_mode

        self.conversion_stats = {
            'total_rows': 0,
            'successful': 0,
            'failed': 0,
            'warnings': 0
        }

        self.conversion_errors: List[Dict[str, Any]] = []
        self.conversion_warnings: List[Dict[str, Any]] = []

    @abstractmethod
    def parse_sequence(self, raw_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract canonical backbone sequence.

        Args:
            raw_data: Raw row data

        Returns:
            Canonical sequence (standard amino acids only) or None if failed
        """
        pass

    @abstractmethod
    def parse_edits(
        self,
        raw_data: Dict[str, Any],
        sequence: str,
        sample_id: str
    ) -> List[Edit]:
        """
        Parse chemical modifications into Edit objects.

        Args:
            raw_data: Raw row data
            sequence: Canonical sequence
            sample_id: Sample identifier

        Returns:
            List of Edit objects
        """
        pass

    @abstractmethod
    def parse_label(self, raw_data: Dict[str, Any]) -> Tuple[float, str, str]:
        """
        Extract experimental label.

        Args:
            raw_data: Raw row data

        Returns:
            (label_value, label_type, label_unit)
        """
        pass

    @abstractmethod
    def parse_assay_metadata(self, raw_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Extract assay information.

        Args:
            raw_data: Raw row data

        Returns:
            (assay_type, assay_metadata)
        """
        pass

    def create_sample_id(self, row_index: int) -> str:
        """
        Generate unique sample ID.

        Args:
            row_index: Row index in source file

        Returns:
            Unique sample ID
        """
        prefix = self.dataset_name.upper().replace(' ', '_').replace('-', '_')
        return f"{prefix}_{row_index:06d}"

    def classify_anchor_status(self, edits: List[Edit]) -> AnchorStatus:
        """
        Determine overall anchor status from edits.

        Args:
            edits: List of Edit objects

        Returns:
            AnchorStatus enum value
        """
        if not edits:
            return AnchorStatus.NO_EDITS

        anchor_kinds = {edit.anchor_kind for edit in edits}

        if anchor_kinds == {AnchorKind.EXPLICIT}:
            return AnchorStatus.EXPLICIT_ANCHOR
        elif AnchorKind.AMBIGUOUS in anchor_kinds:
            return AnchorStatus.NOT_RESOLVABLE
        elif AnchorKind.INFERRED in anchor_kinds or AnchorKind.GLOBAL in anchor_kinds:
            return AnchorStatus.WEAKLY_INFERABLE
        else:
            return AnchorStatus.NOT_RESOLVABLE

    def create_provenance(
        self,
        source_file: str,
        row_index: int,
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create provenance metadata.

        Args:
            source_file: Source CSV filename
            row_index: Row index
            raw_data: Raw row data

        Returns:
            Provenance dictionary
        """
        return {
            'source_file': source_file,
            'source_row_index': row_index,
            'parser_version': self.parser_version,
            'parsing_date': datetime.now().isoformat(),
            'raw_data_sample': {
                k: str(v)[:100] if v else None
                for k, v in list(raw_data.items())[:5]  # First 5 fields
            }
        }

    def convert_row(
        self,
        row_data: Dict[str, Any],
        row_index: int,
        source_file: str
    ) -> Optional[PEMSample]:
        """
        Convert a single row to PEMSample.

        Args:
            row_data: Raw row data
            row_index: Row index
            source_file: Source filename

        Returns:
            PEMSample or None if conversion failed
        """
        sample_id = self.create_sample_id(row_index)

        try:
            # Parse sequence
            sequence = self.parse_sequence(row_data)
            if not sequence:
                raise ValueError("Failed to parse sequence")

            # Parse edits
            edits = self.parse_edits(row_data, sequence, sample_id)

            # Parse label
            label, label_type, label_unit = self.parse_label(row_data)

            # Parse assay
            assay_type, assay_metadata = self.parse_assay_metadata(row_data)

            # Classify anchor status
            anchor_status = self.classify_anchor_status(edits)

            # Create provenance
            provenance = self.create_provenance(source_file, row_index, row_data)

            # Create sample
            sample = PEMSample(
                sample_id=sample_id,
                dataset=self.dataset_name,
                sequence=sequence,
                label=label,
                label_type=label_type,
                label_unit=label_unit,
                assay_type=assay_type,
                assay_metadata=assay_metadata,
                edits=edits,
                anchor_status=anchor_status,
                provenance=provenance,
                quality_flags=[]
            )

            self.conversion_stats['successful'] += 1
            return sample

        except Exception as e:
            self.conversion_stats['failed'] += 1
            self.conversion_errors.append({
                'sample_id': sample_id,
                'row_index': row_index,
                'error': str(e),
                'error_type': type(e).__name__
            })
            return None

    def convert_dataframe(
        self,
        df: pd.DataFrame,
        source_file: str
    ) -> List[PEMSample]:
        """
        Convert entire DataFrame to PEMSamples.

        Args:
            df: Source DataFrame
            source_file: Source filename

        Returns:
            List of successfully converted PEMSample objects
        """
        self.conversion_stats['total_rows'] = len(df)

        samples = []

        for idx, row in df.iterrows():
            row_data = row.to_dict()
            sample = self.convert_row(row_data, idx, source_file)

            if sample:
                samples.append(sample)

        return samples

    def get_conversion_report(self) -> Dict[str, Any]:
        """
        Generate conversion statistics report.

        Returns:
            Dictionary with conversion statistics
        """
        return {
            'dataset': self.dataset_name,
            'parser_version': self.parser_version,
            'stats': self.conversion_stats,
            'success_rate': self.conversion_stats['successful'] / max(self.conversion_stats['total_rows'], 1),
            'errors': self.conversion_errors,
            'warnings': self.conversion_warnings
        }

    def save_conversion_report(self, output_path: Path):
        """
        Save conversion report to JSON file.

        Args:
            output_path: Output file path
        """
        import json

        report = self.get_conversion_report()

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
