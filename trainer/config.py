"""Pydantic configuration models for the LoRA training project."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingSettings(BaseModel):
    """Runtime training configuration loaded from ``training.yaml``.

    Attributes:
        base_model_name: Hugging Face model identifier for the frozen base model.
        output_root: Directory where training artifacts are written.
        adapter_output_root: Directory where LoRA adapters are written.
        log_root: Directory used for TensorBoard and structured logs.
        train_dataset_path: Relative dataset path for training.
        validation_dataset_path: Relative dataset path for validation.
        test_dataset_path: Relative dataset path for evaluation.
        max_seq_length: Maximum tokenizer sequence length.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Per-device train batch size.
        per_device_eval_batch_size: Per-device eval batch size.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        learning_rate: Optimizer learning rate.
        weight_decay: Weight decay factor.
        warmup_ratio: Warmup ratio for scheduler.
        logging_steps: Logging frequency.
        evaluation_strategy: Evaluation scheduling strategy.
        eval_steps: Evaluation frequency when using step evaluation.
        save_strategy: Checkpoint scheduling strategy.
        save_steps: Checkpoint frequency when using step saving.
        save_total_limit: Maximum number of checkpoints to keep.
        seed: Random seed.
        bf16: Whether to use BF16.
        fp16: Whether to use FP16.
        gradient_checkpointing: Whether gradient checkpointing is enabled.
        max_grad_norm: Gradient clipping value.
        lr_scheduler_type: Learning-rate scheduler type.
        report_to: Reporting integrations.
        generate_during_evaluation: Whether to run generation-based metrics.
        generation_max_new_tokens: Max new tokens during evaluation generation.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    base_model_name: str
    output_root: Path
    adapter_output_root: Path
    log_root: Path
    train_dataset_path: Path
    validation_dataset_path: Path
    test_dataset_path: Path
    max_seq_length: int = 1024
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    evaluation_strategy: Literal['no', 'steps', 'epoch'] = 'steps'
    eval_steps: int = 50
    save_strategy: Literal['no', 'steps', 'epoch'] = 'steps'
    save_steps: int = 50
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = 'cosine'
    report_to: list[str] = Field(default_factory=lambda: ['tensorboard'])
    generate_during_evaluation: bool = True
    generation_max_new_tokens: int = 256

    @model_validator(mode='after')
    def validate_precision(self) -> 'TrainingSettings':
        """Prevent invalid mixed precision combinations."""
        if self.bf16 and self.fp16:
            raise ValueError('Only one of bf16 or fp16 can be enabled.')
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> 'TrainingSettings':
        """Build settings from a YAML file path."""
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
        return cls.model_validate(payload)


class LoRASettings(BaseModel):
    """LoRA configuration loaded from ``lora.yaml``."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    bias: str
    task_type: str
    modules_to_save: list[str] = Field(default_factory=list)
    inference_mode: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> 'LoRASettings':
        """Build LoRA settings from a YAML file path."""
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
        return cls.model_validate(payload)
