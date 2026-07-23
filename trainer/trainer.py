"""Training orchestration for LoRA fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from trainer.adapter_manager import AdapterManager
from trainer.config import LoRASettings, TrainingSettings
from trainer.dataset_loader import DatasetLoader
from trainer.evaluator import Evaluator
from trainer.formatter import PromptFormatter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Paths and summaries produced by a training run."""

    run_name: str
    output_dir: Path
    adapter_dir: Path
    log_dir: Path
    training_metrics_path: Path
    evaluation_metrics_path: Path
    summary_path: Path


class LoRATrainingOrchestrator:
    """Coordinate dataset loading, model preparation, training, and evaluation."""

    def __init__(
        self,
        project_root: Path,
        training_settings: TrainingSettings,
        lora_settings: LoRASettings,
        dataset_loader: DatasetLoader,
        formatter: PromptFormatter,
        evaluator: Evaluator,
        adapter_manager: AdapterManager,
    ) -> None:
        self._project_root = project_root
        self._training_settings = training_settings
        self._lora_settings = lora_settings
        self._dataset_loader = dataset_loader
        self._formatter = formatter
        self._evaluator = evaluator
        self._adapter_manager = adapter_manager

    def run(self, dataset_name: str) -> TrainingArtifacts:
        """Execute the full training and evaluation workflow."""
        set_seed(self._training_settings.seed)
        run_paths = self._adapter_manager.training_run_paths(dataset_name)
        logger.info('Starting training run dataset=%s output_dir=%s', dataset_name, run_paths['output_dir'])

        tokenizer = self._load_tokenizer()
        raw_datasets = self._dataset_loader.load_splits(
            self._training_settings.train_dataset_path,
            self._training_settings.validation_dataset_path,
            self._training_settings.test_dataset_path,
        )
        tokenized = DatasetDict(
            {
                split_name: self._prepare_split(dataset, tokenizer)
                for split_name, dataset in raw_datasets.items()
            }
        )

        model = self._load_model()
        trainer = self._build_trainer(model, tokenizer, tokenized, run_paths)
        train_result = trainer.train()
        trainer.save_model(str(run_paths['output_dir']))
        model.save_pretrained(str(run_paths['adapter_dir']))
        tokenizer.save_pretrained(str(run_paths['adapter_dir']))

        validation_loss = self._evaluator.evaluate_loss(trainer, tokenized['validation'], 'validation')
        test_loss = self._evaluator.evaluate_loss(trainer, tokenized['test'], 'test')
        generation_eval = self._evaluator.evaluate_predictions(
            self._training_settings.base_model_name,
            str(run_paths['adapter_dir']),
            raw_datasets['test'],
            tokenizer,
            'test_generation',
        )

        training_metrics = {
            'train_runtime': train_result.metrics.get('train_runtime'),
            'train_loss': train_result.metrics.get('train_loss'),
            'train_samples': len(raw_datasets['train']),
            'validation_loss': validation_loss.loss,
            'test_loss': test_loss.loss,
        }
        evaluation_metrics = {
            'validation_metrics': validation_loss.metrics,
            'test_metrics': test_loss.metrics,
            'generation_metrics': generation_eval.metrics,
        }

        training_metrics_path = run_paths['output_dir'] / 'training_metrics.json'
        evaluation_metrics_path = run_paths['output_dir'] / 'evaluation_metrics.json'
        summary_path = run_paths['output_dir'] / 'training_summary.txt'
        self._adapter_manager.write_json(training_metrics_path, training_metrics)
        self._adapter_manager.write_json(evaluation_metrics_path, evaluation_metrics)
        self._adapter_manager.write_summary(
            summary_path,
            self._build_summary(dataset_name, run_paths, training_metrics, generation_eval.metrics),
        )

        logger.info('Training run completed dataset=%s adapter_dir=%s', dataset_name, run_paths['adapter_dir'])
        return TrainingArtifacts(
            run_name=run_paths['run_name'].name,
            output_dir=run_paths['output_dir'],
            adapter_dir=run_paths['adapter_dir'],
            log_dir=run_paths['log_dir'],
            training_metrics_path=training_metrics_path,
            evaluation_metrics_path=evaluation_metrics_path,
            summary_path=summary_path,
        )

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Load tokenizer for the configured base model."""
        tokenizer = AutoTokenizer.from_pretrained(self._training_settings.base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        return tokenizer

    def _load_model(self) -> PeftModel:
        """Load the frozen base model and attach a trainable LoRA adapter."""
        model = AutoModelForCausalLM.from_pretrained(
            self._training_settings.base_model_name,
            torch_dtype=torch.bfloat16 if self._training_settings.bf16 else None,
        )
        if self._training_settings.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        lora_config = LoraConfig(
            r=self._lora_settings.r,
            lora_alpha=self._lora_settings.alpha,
            lora_dropout=self._lora_settings.dropout,
            target_modules=self._lora_settings.target_modules,
            bias=self._lora_settings.bias,
            task_type=self._lora_settings.task_type,
            modules_to_save=self._lora_settings.modules_to_save or None,
            inference_mode=self._lora_settings.inference_mode,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def _prepare_split(self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        """Format and tokenize one dataset split."""
        formatted = dataset.map(self._formatter.format_sample)

        def tokenize_fn(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            prompts = batch['prompt']
            completions = batch['completion']
            encoded_texts: list[list[int]] = []
            attention_masks: list[list[int]] = []
            labels: list[list[int]] = []
            for prompt, completion in zip(prompts, completions):
                prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
                completion_ids = tokenizer(completion, add_special_tokens=False)['input_ids']
                input_ids = (prompt_ids + completion_ids + [tokenizer.eos_token_id])[: self._training_settings.max_seq_length]
                prompt_length = min(len(prompt_ids), len(input_ids))
                label_ids = [-100] * prompt_length + input_ids[prompt_length:]
                encoded_texts.append(input_ids)
                attention_masks.append([1] * len(input_ids))
                labels.append(label_ids)
            return {'input_ids': encoded_texts, 'attention_mask': attention_masks, 'labels': labels}

        tokenized = formatted.map(tokenize_fn, batched=True, remove_columns=formatted.column_names)
        logger.info('Prepared tokenized split samples=%s', len(tokenized))
        return tokenized

    def _build_trainer(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        datasets: DatasetDict,
        run_paths: dict[str, Path],
    ) -> Trainer:
        """Create the Hugging Face trainer configured for LoRA fine-tuning."""
        args = TrainingArguments(
            output_dir=str(run_paths['output_dir']),
            overwrite_output_dir=True,
            num_train_epochs=self._training_settings.num_train_epochs,
            per_device_train_batch_size=self._training_settings.per_device_train_batch_size,
            per_device_eval_batch_size=self._training_settings.per_device_eval_batch_size,
            gradient_accumulation_steps=self._training_settings.gradient_accumulation_steps,
            learning_rate=self._training_settings.learning_rate,
            weight_decay=self._training_settings.weight_decay,
            warmup_ratio=self._training_settings.warmup_ratio,
            logging_steps=self._training_settings.logging_steps,
            evaluation_strategy=self._training_settings.evaluation_strategy,
            eval_steps=self._training_settings.eval_steps,
            save_strategy=self._training_settings.save_strategy,
            save_steps=self._training_settings.save_steps,
            save_total_limit=self._training_settings.save_total_limit,
            seed=self._training_settings.seed,
            bf16=self._training_settings.bf16,
            fp16=self._training_settings.fp16,
            gradient_checkpointing=self._training_settings.gradient_checkpointing,
            max_grad_norm=self._training_settings.max_grad_norm,
            lr_scheduler_type=self._training_settings.lr_scheduler_type,
            report_to=self._training_settings.report_to,
            logging_dir=str(run_paths['log_dir']),
            load_best_model_at_end=self._training_settings.evaluation_strategy != 'no',
            metric_for_best_model='eval_loss',
        )
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
        return Trainer(
            model=model,
            args=args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            tokenizer=tokenizer,
            data_collator=collator,
        )

    def _build_summary(
        self,
        dataset_name: str,
        run_paths: dict[str, Path],
        training_metrics: dict[str, Any],
        generation_metrics: dict[str, Any],
    ) -> str:
        """Build a human-readable summary of one training run."""
        return '\n'.join(
            [
                f'Dataset: {dataset_name}',
                f'Base Model: {self._training_settings.base_model_name}',
                f'Run Name: {run_paths["run_name"].name}',
                f'Adapter Dir: {run_paths["adapter_dir"]}',
                f'Train Loss: {training_metrics.get("train_loss")}',
                f'Validation Loss: {training_metrics.get("validation_loss")}',
                f'Test Loss: {training_metrics.get("test_loss")}',
                f'Entity Accuracy: {generation_metrics.get("entity_accuracy")}',
                f'Operation Accuracy: {generation_metrics.get("operation_accuracy")}',
                f'Domain Accuracy: {generation_metrics.get("domain_accuracy")}',
                f'Service Accuracy: {generation_metrics.get("service_accuracy")}',
                f'Parameter Extraction Accuracy: {generation_metrics.get("parameter_extraction_accuracy")}',
                f'Overall Accuracy: {generation_metrics.get("overall_accuracy")}',
            ]
        )
