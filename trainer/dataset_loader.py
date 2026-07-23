"""Dataset loading utilities for LoRA fine-tuning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from trainer.exceptions import DatasetValidationError

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and validate JSONL datasets for training, validation, and test.

    The loader preserves the master dataset as read-only source material and loads
    only the split datasets required for fine-tuning and evaluation.
    """

    REQUIRED_TOP_LEVEL_FIELDS = {'instruction', 'input', 'output'}
    REQUIRED_OUTPUT_FIELDS = {
        'domain',
        'service',
        'entity',
        'operation',
        'requiresTool',
        'parameters',
        'responseType',
        'confidence',
    }

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def load_splits(self, train_path: Path, validation_path: Path, test_path: Path) -> DatasetDict:
        """Load the train, validation, and test dataset splits.

        Args:
            train_path: Relative path to the training dataset.
            validation_path: Relative path to the validation dataset.
            test_path: Relative path to the test dataset.

        Returns:
            A ``DatasetDict`` containing ``train``, ``validation``, and ``test`` splits.
        """
        datasets = {
            'train': self.load_jsonl(train_path),
            'validation': self.load_jsonl(validation_path),
            'test': self.load_jsonl(test_path),
        }
        logger.info(
            'Loaded dataset splits train=%s validation=%s test=%s',
            len(datasets['train']),
            len(datasets['validation']),
            len(datasets['test']),
        )
        return DatasetDict(datasets)

    def load_jsonl(self, relative_path: Path) -> Dataset:
        """Load and validate a JSONL file into a Hugging Face dataset."""
        file_path = self._project_root / relative_path
        if not file_path.exists():
            raise DatasetValidationError(f'Dataset file not found: {file_path}')

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line_number, line in enumerate(file_path.read_text(encoding='utf-8').splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f'Invalid JSON at {file_path}:{line_number}') from exc
            self._validate_record(record, file_path, line_number)
            serialized = json.dumps(record, sort_keys=True, ensure_ascii=False)
            if serialized in seen:
                raise DatasetValidationError(f'Duplicate record detected in {file_path}:{line_number}')
            seen.add(serialized)
            rows.append(record)

        if not rows:
            raise DatasetValidationError(f'Dataset file is empty: {file_path}')

        logger.info('Validated dataset file path=%s samples=%s', file_path, len(rows))
        return Dataset.from_list(rows)

    def _validate_record(self, record: dict[str, Any], file_path: Path, line_number: int) -> None:
        """Validate the planner dataset schema for a single record."""
        if set(record.keys()) != self.REQUIRED_TOP_LEVEL_FIELDS:
            raise DatasetValidationError(
                f'Invalid top-level schema at {file_path}:{line_number}. '
                f'Expected {self.REQUIRED_TOP_LEVEL_FIELDS}, got {set(record.keys())}'
            )
        output = record.get('output')
        if not isinstance(output, dict) or set(output.keys()) != self.REQUIRED_OUTPUT_FIELDS:
            raise DatasetValidationError(
                f'Invalid output schema at {file_path}:{line_number}. '
                f'Expected {self.REQUIRED_OUTPUT_FIELDS}, got {set(output.keys()) if isinstance(output, dict) else type(output)}'
            )
