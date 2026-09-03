"""JSONL serialization utilities for the final PEM dataset."""

import json
from pathlib import Path
from typing import List

from .pem_schema import PEMSample, sample_to_dict


def save_jsonl(samples: List[PEMSample], output_path: Path) -> int:
    """
    Save samples to JSONL format.

    Args:
        samples: List of PEMSample objects
        output_path: Output file path (.jsonl)

    Returns:
        Number of samples written
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            sample_dict = sample_to_dict(sample)
            f.write(json.dumps(sample_dict, default=str) + '\n')

    return len(samples)


def load_jsonl(input_path: Path) -> List[PEMSample]:
    """
    Load samples from JSONL format.

    Args:
        input_path: Input file path (.jsonl)

    Returns:
        List of PEMSample objects
    """
    samples = []

    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                sample_dict = json.loads(line)
                sample = PEMSample(**sample_dict)
                samples.append(sample)

    return samples


def load_samples(input_path: Path) -> List[PEMSample]:
    """Load PEM samples from the release JSONL format."""
    input_path = Path(input_path)

    if input_path.suffix != '.jsonl':
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .jsonl")
    return load_jsonl(input_path)


# Convenience aliases
save_samples_jsonl = save_jsonl
load_samples_jsonl = load_jsonl
