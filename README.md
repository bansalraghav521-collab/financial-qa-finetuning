# Financial Q&A Fine-Tuning with Qwen3-1.7B and QLoRA

This repository contains a complete assignment submission for fine-tuning a small open-source language model on the Financial Q&A task. The project uses `Qwen/Qwen3-1.7B` with QLoRA, trains on a curated subset of `gbharti/finance-alpaca`, and demonstrates measurable improvement over the base model on ROUGE, BLEU, task-specific financial accuracy, and qualitative evaluation.

It is organized to satisfy the assignment deliverables:

| Assignment requirement | Where it is covered |
|---|---|
| Training pipeline | [`src/`](./src) |
| Config files | [`src/config/qlora_config.yaml`](./src/config/qlora_config.yaml) |
| Dataset sample | [`data/sample_data.jsonl`](./data/sample_data.jsonl) |
| Demo video | [`DEMO_VIDEO`](./DEMO_VIDEO) |
| Model artifacts | [Hugging Face adapters](https://huggingface.co/IItianRaghav/qwen3-finance-qlora) |
| Dataset design | [`DATASET.md`](./DATASET.md) |
| Training details | [`TRAINING.md`](./TRAINING.md) |
| Evaluation results | [`EVALUATION.md`](./EVALUATION.md) |
| Failure cases | [`EVALUATION.md`](./EVALUATION.md#failure-cases) |
| Architecture / diagrams | [`ARCHITECTURE.md`](./ARCHITECTURE.md), [`ARCHITECTURE_DIAGRAM.md`](./ARCHITECTURE_DIAGRAM.md) |

## 1. Task

**Chosen task:** Financial Q&A

- Input: a financial question
- Output: an accurate, grounded financial answer

Example:

```text
Input: What is the difference between a stock and a bond?
Output: A stock represents ownership in a company, while a bond is a debt
instrument issued by a company or government. Stocks offer upside through
growth and dividends, while bonds typically provide fixed interest payments
and lower risk.
```

## 2. Model Selection

**Base model:** `Qwen/Qwen3-1.7B`

### Why this model?

- Small enough to fine-tune on a free Kaggle T4 GPU with 16 GB VRAM
- Strong instruction-following baseline, which makes it a good fit for Financial Q&A
- Works well with 4-bit QLoRA, allowing low-cost training without full fine-tuning
- Fast enough to support short-burst experimentation during hyperparameter search

### Trade-offs

- A 1.7B model is much cheaper and faster to train than larger models such as Mistral 7B
- The trade-off is lower ceiling on complex reasoning and long multi-step calculations
- For this task, that trade-off is acceptable because most target answers are short, factual, and concept-focused

## 3. Fine-Tuning Approach

This project uses **QLoRA**, which combines:

- 4-bit NF4 quantization for the frozen base model
- LoRA adapters for parameter-efficient training

### Configuration

- Quantization: 4-bit NF4 with `bitsandbytes`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Trainer: `trl.SFTTrainer`

This approach was chosen over full fine-tuning because it fits on the available hardware while still delivering strong domain adaptation performance.

## 4. Dataset

**Source:** [`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca)

- Original dataset size: about 68,000 samples
- Curated subset used for training: `1,500` samples
- Task format: instruction tuning

### Data preparation

The dataset pipeline in [`src/prepare_data.py`](./src/prepare_data.py) performs:

- filtering for output length between 20 and 300 words
- removal of HTML / broken markup samples
- removal of short or low-quality instructions
- near-duplicate filtering using fuzzy matching
- balancing across 7 financial topic areas: interest rates, stocks, personal finance, economic indicators, financial instruments, tax, and risk / portfolio

### Prompt format

Each sample is converted to Alpaca-style instruction tuning text. The old placeholder-style template looked empty on GitHub, so this section now shows a real example:

```text
### Instruction:
What is compound interest?

### Response:
Compound interest is interest calculated on both the principal and the accumulated
interest from previous periods. Formula: A = P(1 + r/n)^(nt)
```

### Splits

- Train: `80%`
- Validation: `10%`
- Evaluation: `10%`

### Sample data

Reviewers can inspect a small sample file here:

- [`data/sample_data.jsonl`](./data/sample_data.jsonl)

Full dataset notes are documented in [`DATASET.md`](./DATASET.md).

## 5. Training Pipeline

The training pipeline is implemented in [`src/train.py`](./src/train.py).

### Tech stack

- Python
- `transformers`
- `peft`
- `bitsandbytes`
- `trl`
- `datasets`
- `rouge-score`
- `sacrebleu`
- Hugging Face Hub
- Kaggle Notebooks

### Components

- Tokenizer: `AutoTokenizer`
- Padding: right-padding with EOS as pad token
- Base model loading: `AutoModelForCausalLM`
- Quantization: `BitsAndBytesConfig`
- Adapter injection: `peft.LoraConfig`
- Training loop: `trl.SFTTrainer`

### Key hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate | `2e-4` |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Scheduler | cosine |
| Warmup ratio | 0.03 |
| Max sequence length | 512 |
| Optimizer | `paged_adamw_32bit` |

### Training platform

- Platform: Kaggle Notebooks
- GPU: NVIDIA T4
- VRAM: 16 GB
- Precision: FP16

Additional training details are documented in [`TRAINING.md`](./TRAINING.md).

## 6. Evaluation

The assignment requires comparison between the base model and the fine-tuned model using BLEU, ROUGE, task-specific evaluation, and qualitative comparison. That is fully documented in [`EVALUATION.md`](./EVALUATION.md).

### Evaluation summary

| Metric | Base Model | Fine-tuned Model | Improvement |
|---|---|---|---|
| ROUGE-1 | 0.1808 | 0.2813 | +55% |
| ROUGE-2 | 0.0874 | 0.1417 | +62% |
| ROUGE-L | 0.1687 | 0.2336 | +38% |
| BLEU | 1.1640 | 2.6316 | +126% |
| Financial accuracy | 60% | 85% | +25 points |
| Qualitative avg (1-5) | 3.8 | 4.2 | +11% |

### What is included in the evaluation report?

- Base model vs fine-tuned model comparison
- ROUGE-1, ROUGE-2, and ROUGE-L
- BLEU
- Task-specific financial accuracy
- 10-question qualitative comparison
- failure cases and limitations

See [`EVALUATION.md`](./EVALUATION.md) for the full report.

## 7. Bonus / X-Factor

This submission includes two bonus-worthy additions beyond the minimum requirement:

### QLoRA / quantization

- 4-bit NF4 quantization makes the full pipeline trainable on a free GPU
- this directly aligns with the optional "quantization" bonus mentioned in the assignment

### Autonomous autoresearch loop

The repository includes [`src/autoresearch_loop.py`](./src/autoresearch_loop.py), inspired by Karpathy's autoresearch concept.

Instead of manually choosing one hyperparameter set, the loop:

1. defines a validation metric
2. samples a candidate config
3. runs a short-burst training experiment
4. scores it
5. keeps or discards the config
6. repeats

Search dimensions:

- LoRA rank: `8`, `16`, `32`
- Learning rate: `1e-4`, `2e-4`, `5e-4`
- Prompt format: `alpaca`, `chat`, `minimal`

This is documented further in [`TRAINING.md`](./TRAINING.md) and visualized in [`ARCHITECTURE_DIAGRAM.md`](./ARCHITECTURE_DIAGRAM.md).

## 8. Model Artifacts

- Base model: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)
- Fine-tuned adapters: [IItianRaghav/qwen3-finance-qlora](https://huggingface.co/IItianRaghav/qwen3-finance-qlora)

The training script saves adapters locally and can also push them to Hugging Face Hub when `HF_TOKEN` is available.

## 9. Demo

- Demo placeholder: [`DEMO_VIDEO`](./DEMO_VIDEO)

This file is included to satisfy the assignment's "Demo (Video or Loom)" deliverable. The intended demo is a before-vs-after comparison showing base-model outputs against fine-tuned outputs on representative Financial Q&A prompts.

## 10. Repository Structure

```text
repo/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── inference.py
│   ├── autoresearch_loop.py
│   └── config/
│       └── qlora_config.yaml
├── data/
│   ├── sample_data.jsonl
│   ├── train.jsonl
│   ├── val.jsonl
│   └── eval.jsonl
├── evaluation/
│   ├── evaluate.py
│   └── results/
├── README.md
├── ARCHITECTURE.md
├── ARCHITECTURE_DIAGRAM.md
├── DATASET.md
├── TRAINING.md
├── EVALUATION.md
├── DEMO_VIDEO
└── requirements.txt
```

## 11. Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with at least 12 GB VRAM
- Tested on Kaggle T4 (16 GB)

### Install

```bash
pip install -r requirements.txt
```

Optional if you want to push adapters to Hugging Face:

```bash
export HF_TOKEN=your_token_here
export HF_USERNAME=your_username
```

## 12. How to Run

### Step 1: Prepare the dataset

```bash
python src/prepare_data.py --output_dir ./data --seed 42 --n_samples 1500
```

### Step 2: Run the autoresearch loop (optional)

```bash
python src/autoresearch_loop.py \
  --train_data ./data/train.jsonl \
  --val_data ./data/val.jsonl \
  --n_iter 15 \
  --output_config ./src/config/best_config.yaml
```

### Step 3: Train the model

```bash
python src/train.py \
  --config ./src/config/qlora_config.yaml \
  --output_dir ./outputs
```

### Step 4: Evaluate base vs fine-tuned

```bash
python evaluation/evaluate.py \
  --base_model Qwen/Qwen3-1.7B \
  --adapter_path ./outputs/adapters \
  --eval_data ./data/eval.jsonl \
  --output_dir ./evaluation/results
```

### Step 5: Run inference

```bash
python src/inference.py \
  --base_model Qwen/Qwen3-1.7B \
  --adapter_path ./outputs/adapters \
  --question "What is compound interest?"
```

If your local `transformers` build requires it for Qwen, add `--trust_remote_code` to training, evaluation, autoresearch, and inference commands.

## 13. Documentation

| File | Purpose |
|---|---|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | system design and component breakdown |
| [`ARCHITECTURE_DIAGRAM.md`](./ARCHITECTURE_DIAGRAM.md) | ASCII pipeline diagrams |
| [`DATASET.md`](./DATASET.md) | dataset source, cleaning, formatting, split design |
| [`TRAINING.md`](./TRAINING.md) | QLoRA config, hyperparameters, training environment |
| [`EVALUATION.md`](./EVALUATION.md) | metrics, qualitative comparison, failure cases |
| [`DEMO_VIDEO`](./DEMO_VIDEO) | demo / Loom placeholder |

## 14. Notes

- The repository is intended to be runnable from the instructions in this README
- The report components required by the assignment are broken out into separate Markdown files for clarity
- The fine-tuned model improves on the base model across all required evaluation categories
