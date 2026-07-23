"""Evaluate a saved LoRA adapter against the test dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_import_path() -> None:
    """Keep the project importable without shadowing installed packages."""
    normalized_path: list[str] = []
    for entry in sys.path:
        resolved = Path(entry or Path.cwd()).resolve()
        if resolved == PROJECT_ROOT:
            continue
        normalized_path.append(entry)
    sys.path[:] = normalized_path
    sys.path.append(str(PROJECT_ROOT))


configure_import_path()

from transformers import AutoTokenizer

from trainer.config import TrainingSettings
from trainer.dataset_loader import DatasetLoader
from trainer.evaluator import Evaluator
from trainer.formatter import PromptFormatter


def configure_logging() -> None:
    """Configure console logging for evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate a saved ERP planner LoRA adapter.')
    parser.add_argument('--adapter-path', required=True, help='Path to the saved LoRA adapter directory.')
    parser.add_argument('--training-config', default='configs/training.yaml', help='Path to training.yaml.')
    return parser.parse_args()


def main() -> int:
    """Run generation-based evaluation on the test split."""
    configure_logging()
    args = parse_args()
    project_root = PROJECT_ROOT
    training_settings = TrainingSettings.from_yaml(project_root / args.training_config)
    loader = DatasetLoader(project_root)
    datasets = loader.load_splits(
        training_settings.train_dataset_path,
        training_settings.validation_dataset_path,
        training_settings.test_dataset_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(training_settings.base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    evaluator = Evaluator(PromptFormatter(), max_new_tokens=training_settings.generation_max_new_tokens)
    result = evaluator.evaluate_predictions(
        training_settings.base_model_name,
        args.adapter_path,
        datasets['test'],
        tokenizer,
        'test_generation',
    )
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
