"""Utilities for saving adapters and training artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AdapterManager:
    """Manage adapter, metrics, and summary output paths."""

    def __init__(self, project_root: Path, output_root: Path, adapter_root: Path, log_root: Path) -> None:
        self._project_root = project_root
        self._output_root = project_root / output_root
        self._adapter_root = project_root / adapter_root
        self._log_root = project_root / log_root
        for path in (self._output_root, self._adapter_root, self._log_root):
            path.mkdir(parents=True, exist_ok=True)

    def training_run_paths(self, dataset_name: str) -> dict[str, Path]:
        """Create a timestamped set of output paths for a training run."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_name = f'{dataset_name}-{timestamp}'
        paths = {
            'run_name': Path(run_name),
            'output_dir': self._output_root / run_name,
            'adapter_dir': self._adapter_root / run_name,
            'log_dir': self._log_root / run_name,
        }
        for key, path in paths.items():
            if key != 'run_name':
                path.mkdir(parents=True, exist_ok=True)
        return paths

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Write a JSON artifact with UTF-8 encoding."""
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    def write_summary(self, path: Path, summary: str) -> None:
        """Write a plain-text summary artifact."""
        path.write_text(summary, encoding='utf-8')
