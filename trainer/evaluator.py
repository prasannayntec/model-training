"""Evaluation utilities for LoRA fine-tuning outputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, Trainer

from trainer.formatter import PromptFormatter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Container for validation/test evaluation results."""

    split_name: str
    loss: float | None
    metrics: dict[str, float]


class Evaluator:
    """Evaluate LoRA adapters with loss and generation-based metrics."""

    def __init__(self, formatter: PromptFormatter, max_new_tokens: int = 256) -> None:
        self._formatter = formatter
        self._max_new_tokens = max_new_tokens
        self._exact_match = evaluate.load('exact_match')

    def evaluate_loss(self, trainer: Trainer, eval_dataset: Dataset, split_name: str) -> EvaluationResult:
        """Compute loss-oriented evaluation using the Hugging Face trainer."""
        result = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=split_name)
        loss = result.get(f'{split_name}_loss')
        return EvaluationResult(split_name=split_name, loss=loss, metrics=result)

    @torch.no_grad()
    def evaluate_predictions(
        self,
        base_model_name: str,
        adapter_path: str,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        split_name: str,
    ) -> EvaluationResult:
        """Compute generation-based accuracy metrics on a dataset split."""
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        rows: list[dict[str, Any]] = []
        exact_matches: list[dict[str, str]] = []
        for sample in dataset:
            formatted = self._formatter.format_sample(sample)
            prompt = formatted['prompt']
            expected = sample['output']
            predicted = self._generate_json(model, tokenizer, prompt)
            rows.append(self._compare_outputs(expected, predicted))
            exact_matches.append(
                {
                    'prediction': json.dumps(predicted, ensure_ascii=False, sort_keys=True),
                    'reference': json.dumps(expected, ensure_ascii=False, sort_keys=True),
                }
            )

        frame = pd.DataFrame(rows)
        exact_match = self._exact_match.compute(
            predictions=[item['prediction'] for item in exact_matches],
            references=[item['reference'] for item in exact_matches],
        )['exact_match']
        metrics = {
            'domain_accuracy': float(frame['domain_match'].mean()),
            'service_accuracy': float(frame['service_match'].mean()),
            'entity_accuracy': float(frame['entity_match'].mean()),
            'operation_accuracy': float(frame['operation_match'].mean()),
            'parameter_extraction_accuracy': float(frame['parameters_match'].mean()),
            'overall_accuracy': float(frame['overall_match'].mean()),
            'exact_match': float(exact_match),
            'sample_count': float(len(rows)),
        }
        logger.info('Completed generation evaluation split=%s metrics=%s', split_name, metrics)
        return EvaluationResult(split_name=split_name, loss=None, metrics=metrics)

    def _generate_json(
        self,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response from the fine-tuned model."""
        inputs = tokenizer(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}
        output_ids = model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        return self._extract_json_object(generated)

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        """Extract a JSON object from generated text safely."""
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end < start:
            return {}
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _compare_outputs(self, expected: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
        """Compare predicted and expected planner outputs."""
        return {
            'domain_match': expected.get('domain') == predicted.get('domain'),
            'service_match': expected.get('service') == predicted.get('service'),
            'entity_match': expected.get('entity') == predicted.get('entity'),
            'operation_match': expected.get('operation') == predicted.get('operation'),
            'parameters_match': self._normalize_parameters(expected.get('parameters', {}))
            == self._normalize_parameters(predicted.get('parameters', {})),
            'overall_match': expected == predicted,
        }

    def _normalize_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Normalize parameter dictionaries for comparison."""
        return json.loads(json.dumps(parameters, sort_keys=True, ensure_ascii=False))
