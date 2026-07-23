# Model Training

Independent LoRA fine-tuning project for ERP planner datasets. This project is responsible only for adapter training and evaluation. It does not integrate with AI Runtime, Model Gateway, MCP, or FastAPI.

## Structure

- `configs/training.yaml`: training configuration
- `configs/lora.yaml`: LoRA configuration
- `datasets/academic/master/academic_dataset.jsonl`: immutable source-of-truth dataset
- `datasets/academic/train/train.jsonl`: training split
- `datasets/academic/validation/validation.jsonl`: validation split
- `datasets/academic/test/test.jsonl`: test split
- `trainer/`: dataset loading, formatting, training, evaluation, and adapter management
- `scripts/train.py`: training entrypoint
- `scripts/evaluate_model.py`: evaluation entrypoint
- `outputs/`: training artifacts and summaries
- `adapters/`: saved LoRA adapters
- `logs/`: TensorBoard logs

## Dataset Contract

Each dataset sample must follow this schema:

```json
{
  "instruction": "",
  "input": "",
  "output": {
    "domain": "",
    "service": "",
    "entity": "",
    "operation": "",
    "requiresTool": true,
    "parameters": {},
    "responseType": "",
    "confidence": 0.95
  }
}
```

The master dataset is never modified by training code. Train on `train.jsonl`, validate on `validation.jsonl`, and evaluate on `test.jsonl`.

## Configuration

Update these files to change training behavior:

- `configs/training.yaml`
- `configs/lora.yaml`

All hyperparameters, paths, LoRA settings, reporting, and evaluation settings are loaded from YAML.

## Training

From `model-training/`:

```bash
python scripts/train.py --dataset-name academic
```

This will:

- load train and validation splits
- format prompts for causal LM fine-tuning
- attach a LoRA adapter to the frozen base model
- train and evaluate checkpoints
- save the adapter
- write metrics and a training summary
- emit TensorBoard logs

## Evaluation

```bash
python scripts/evaluate_model.py --adapter-path adapters/<run-name>
```

This reports:

- validation/test loss artifacts from training outputs
- domain accuracy
- service accuracy
- entity accuracy
- operation accuracy
- parameter extraction accuracy
- overall accuracy
- exact JSON match rate

## Notes

- Base model weights are never modified.
- Only LoRA adapters are saved.
- The code is domain-agnostic; future datasets can reuse the same pipeline by changing dataset paths and configs.

