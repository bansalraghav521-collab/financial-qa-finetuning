# Financial Q&A Fine-Tuning — Qwen3-1.7B with QLoRA

## Overview

This project fine-tunes `Qwen/Qwen3-1.7B` for Financial Question Answering using QLoRA. It demonstrates measurable improvement over the base model across ROUGE, BLEU, and qualitative evaluation metrics while remaining lightweight enough to train on a Kaggle T4 GPU.

## Model

- Base model: `Qwen/Qwen3-1.7B`
- Fine-tuning method: QLoRA with 4-bit NF4 quantization and LoRA adapters
- Training platform: Kaggle T4 GPU (16 GB VRAM)
- Hugging Face adapters: [IItianRaghav/qwen3-finance-qlora](https://huggingface.co/IItianRaghav/qwen3-finance-qlora)

## Results

| Metric | Base Model | Fine-tuned | Improvement |
|---|---|---|---|
| ROUGE-1 | 0.1808 | 0.2813 | +55% |
| ROUGE-2 | 0.0874 | 0.1417 | +62% |
| ROUGE-L | 0.1687 | 0.2336 | +38% |
| BLEU | 1.1640 | 2.6316 | +126% |

Full evaluation details are available in [EVALUATION.md](./EVALUATION.md).

## Dataset

The training data comes from the `gbharti/finance-alpaca` dataset on Hugging Face. A curated subset of 1,500 samples was selected and balanced across seven financial topic areas:

- Interest rates
- Stocks
- Personal finance
- Economic indicators
- Financial instruments
- Tax
- Risk and portfolio management

All samples were formatted in Alpaca-style instruction-tuning format before training.

## Tech Stack

- Python
- `transformers`
- `peft`
- `bitsandbytes`
- `trl`
- `datasets`
- `rouge-score`
- `sacrebleu`

## Autoresearch

This project includes an autonomous hyperparameter search loop inspired by Karpathy's autoresearch concept. For the reported tuning pass, 6 experiments were run over LoRA rank (`8`, `16`, `32`) and learning rate (`1e-4`, `2e-4`, `5e-4`) combinations, scored using a combined ROUGE metric. The search confirmed the original configuration of `r=16` and `lr=2e-4` as the best choice.

## Setup

```bash
pip install -r requirements.txt
```

## How To Run

1. Prepare the dataset:

```bash
python src/prepare_data.py --output_dir ./data/ --seed 42
```

2. Train the model:

```bash
python src/train.py --config src/config/qlora_config.yaml --output_dir ./outputs
```

3. Evaluate the model:

```bash
python evaluation/evaluate.py --base_model Qwen/Qwen3-1.7B --adapter_path ./outputs/adapters --eval_data ./data/eval.jsonl --output_dir ./evaluation/results/
```

## Repository Structure

```text
src/                 training pipeline, inference, autoresearch loop
data/                processed dataset splits
evaluation/          evaluation scripts and results
requirements.txt     dependencies
ARCHITECTURE.md      system design
EVALUATION.md        full evaluation results
DATASET.md           dataset design
TRAINING.md          training details
```

## Submission

[Submission Form](https://forms.gle/9iPeUBHKcdHhuSq67)
