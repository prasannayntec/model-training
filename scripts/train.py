"""Train a LoRA adapter for an ERP planner dataset."""

from __future__ import annotations

import argparse
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

from trainer.adapter_manager import AdapterManager
from trainer.config import LoRASettings, TrainingSettings
from trainer.dataset_loader import DatasetLoader
from trainer.evaluator import Evaluator
from trainer.formatter import PromptFormatter
from trainer.trainer import LoRATrainingOrchestrator


def configure_logging() -> None:
    """Configure console logging for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description='Train a LoRA adapter for ERP planner data.')
    parser.add_argument('--dataset-name', default='academic', help='Dataset domain folder name.')
    parser.add_argument('--training-config', default='configs/training.yaml', help='Path to training.yaml.')
    parser.add_argument('--lora-config', default='configs/lora.yaml', help='Path to lora.yaml.')
    return parser.parse_args()


def main() -> int:
    """Run the end-to-end LoRA training workflow."""
    configure_logging()
    args = parse_args()
    project_root = PROJECT_ROOT
    training_settings = TrainingSettings.from_yaml(project_root / args.training_config)
    lora_settings = LoRASettings.from_yaml(project_root / args.lora_config)

    orchestrator = LoRATrainingOrchestrator(
        project_root=project_root,
        training_settings=training_settings,
        lora_settings=lora_settings,
        dataset_loader=DatasetLoader(project_root),
        formatter=PromptFormatter(),
        evaluator=Evaluator(PromptFormatter(), max_new_tokens=training_settings.generation_max_new_tokens),
        adapter_manager=AdapterManager(
            project_root,
            training_settings.output_root,
            training_settings.adapter_output_root,
            training_settings.log_root,
        ),
    )
    artifacts = orchestrator.run(args.dataset_name)
    logging.getLogger(__name__).info('Training complete. Adapter saved to %s', artifacts.adapter_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
