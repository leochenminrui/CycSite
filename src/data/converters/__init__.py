"""Dataset-to-PEM schema converters.

This study uses a single dataset: CycPeptMPDB (PAMPA subset).
Converters for PepMSND and DBAASP are outside the currently supported project
data pipeline. The CycPeptMPDB converter is the supported implementation.
"""

from .base_converter import BaseConverter
from .cycpeptmpdb_converter import CycPeptMPDBConverter

__all__ = [
    'BaseConverter',
    'CycPeptMPDBConverter',
]
