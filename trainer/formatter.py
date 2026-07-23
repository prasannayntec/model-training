"""Prompt formatting utilities for causal language model fine-tuning."""

from __future__ import annotations

import json
from typing import Any


class PromptFormatter:
    """Format planner samples into supervised fine-tuning prompts."""

    SYSTEM_PROMPT = (
        'You are an ERP Planner model. Read the instruction and user request, then '
        'return only the final JSON output object that matches the planner schema.'
    )

    def format_sample(self, sample: dict[str, Any]) -> dict[str, str]:
        """Convert a dataset sample into prompt and completion strings."""
        prompt = (
            f'<|system|>\n{self.SYSTEM_PROMPT}\n'
            f'<|instruction|>\n{sample["instruction"]}\n'
            f'<|input|>\n{sample["input"]}\n'
            '<|output|>\n'
        )
        completion = json.dumps(sample['output'], ensure_ascii=False, sort_keys=True)
        return {'prompt': prompt, 'completion': completion, 'text': prompt + completion}
