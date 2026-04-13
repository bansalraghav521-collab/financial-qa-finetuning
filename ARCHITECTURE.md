# ARCHITECTURE.md — Financial Q&A Fine-Tuning Pipeline

## Overview

This project fine-tunes a small language model (Qwen3 Mini) using QLoRA for the task of **Financial Question Answering**. The pipeline is designed to be reproducible, lightweight (runs on a single T4 GPU), and extensible with an autonomous experimentation loop inspired by Karpathy's `autoresearch` framework.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROJECT PIPELINE                             │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐ │
│  │  Dataset │───▶│  Preprocess  │───▶│  QLoRA Fine-Tuning Loop   │ │
│  │ (finance-│    │  & Format    │    │  (Kaggle T4 GPU)          │ │
│  │ alpaca)  │    │  (instruc.   │    └───────────┬───────────────┘ │
│  └──────────┘    │   tuning     │                │                 │
│                  │   style)     │    ┌───────────▼───────────────┐ │
│                  └──────────────┘    │   LoRA Adapter Weights    │ │
│                                      │   (saved to HuggingFace)  │ │
│                                      └───────────┬───────────────┘ │
│                                                  │                 │
│                                      ┌───────────▼───────────────┐ │
│                                      │     Evaluation Module     │ │
│                                      │  Base vs Fine-tuned Model │ │
│                                      │  ROUGE / Qualitative      │ │
│                                      └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Base Model — Qwen3 Mini

| Property | Detail |
|---|---|
| Model | `Qwen/Qwen3-0.6B` or `Qwen/Qwen3-1.7B` |
| Type | Decoder-only transformer (causal LM) |
| Parameters | ~0.6B–1.7B |
| Quantization | 4-bit NF4 via `bitsandbytes` |
| Source | HuggingFace Hub |

**Why Qwen3 Mini?**
- Small enough to fine-tune on a free Kaggle T4 GPU (16GB VRAM)
- Strong instruction-following baseline out of the box
- Supports 4-bit QLoRA without accuracy collapse
- Efficient domain adaptation at low compute cost — justifiable trade-off between size and performance

---

### 2. Dataset Module (`/data`)

```
data/
├── raw/                  # Original downloaded dataset
├── processed/            # Cleaned, formatted samples
├── train.jsonl           # Training split (80%)
├── val.jsonl             # Validation split (10%)
└── eval.jsonl            # Held-out evaluation set (10%)
```

**Format — Instruction Tuning Style:**
```json
{
  "instruction": "What is compound interest and how is it calculated?",
  "input": "",
  "output": "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Formula: A = P(1 + r/n)^(nt), where P = principal, r = annual interest rate, n = compounding frequency, t = time in years."
}
```

**Source:** `finance-alpaca` from HuggingFace (~68k samples; curated 1500-sample subset used)

---

### 3. Fine-Tuning Module (`/src`)

```
src/
├── prepare_data.py           # Download, clean, format dataset
├── train.py                  # Main QLoRA training script
├── autoresearch_loop.py      # Autonomous hyperparameter search loop
├── inference.py              # Run inference on base vs fine-tuned model
└── config/
    └── qlora_config.yaml     # All hyperparameters in one place
```

**Training Stack:**

| Library | Role |
|---|---|
| `transformers` | Model loading, tokenization, Trainer |
| `peft` | LoRA adapter injection and management |
| `bitsandbytes` | 4-bit quantization (QLoRA) |
| `trl` | SFTTrainer (Supervised Fine-Tuning wrapper) |
| `datasets` | Dataset loading and processing |
| `accelerate` | GPU acceleration backend |

**QLoRA Configuration:**
```yaml
lora_r: 16                      # LoRA rank
lora_alpha: 32                  # Scaling factor
lora_dropout: 0.05
target_modules:                 # Which attention layers receive LoRA adapters
  - q_proj
  - v_proj
  - k_proj
  - o_proj
bits: 4                         # 4-bit quantization
bnb_4bit_compute_dtype: float16
bnb_4bit_quant_type: nf4
```

**Training Hyperparameters:**
```yaml
learning_rate: 2e-4
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
warmup_ratio: 0.03
lr_scheduler_type: cosine
max_seq_length: 512
```

---

### 4. Autonomous Experiment Loop — X Factor

Inspired by Karpathy's `autoresearch` concept. Instead of manually guessing one hyperparameter set, the pipeline runs short experiment bursts, scores each against a defined metric (ROUGE-L), keeps what improves, and discards what doesn't. The best config found then drives the full training run.

```
┌──────────────────────────────────────────────────────┐
│               AUTORESEARCH LOOP                      │
│                                                      │
│   Metric: ROUGE-L on held-out eval.jsonl             │
│         │                                            │
│         ▼                                            │
│   Sample Config (lr, lora_r, prompt_format)          │
│         │                                            │
│         ▼                                            │
│   Train for N steps (short burst ~5 min)             │
│         │                                            │
│         ▼                                            │
│   Score against ROUGE-L                              │
│         │                                            │
│       Better than best?                              │
│      YES ──▶ Save config                             │
│       NO ──▶ Discard config                          │
│         │                                            │
│   Next iteration (repeat ~10–20 times)               │
│         │                                            │
│   Full training with best config found               │
└──────────────────────────────────────────────────────┘
```

**Search space:**
- `lora_r`: [8, 16, 32]
- `learning_rate`: [1e-4, 2e-4, 5e-4]
- `prompt_format`: [alpaca-style, chat-style, minimal]

---

### 5. Evaluation Module (`/evaluation`)

```
evaluation/
├── evaluate.py                  # Runs base vs fine-tuned on eval set
├── results/
│   ├── base_model_outputs.json
│   ├── finetuned_outputs.json
│   └── rouge_scores.json
└── EVALUATION.md                # Full evaluation report
```

---

### 6. Model Artifacts

| Artifact | Location |
|---|---|
| LoRA Adapters | HuggingFace Hub (public repo) |
| Base model (reference) | `Qwen/Qwen3-1.7B` on HuggingFace |
| Tokenizer | Saved alongside adapters |
| Config files | `/src/config/` in GitHub repo |

---

## Data Flow — End to End

```
finance-alpaca (HuggingFace)
        │
        ▼
prepare_data.py
  → filter financial Q&A samples
  → clean + deduplicate
  → format to instruction-tuning style
  → split: train / val / eval
        │
        ▼
autoresearch_loop.py
  → short burst experiments over config search space
  → find best LoRA config via ROUGE-L signal
        │
        ▼
train.py (full training with best config)
  → QLoRA fine-tuning on Kaggle T4 GPU
  → saves LoRA adapters locally + HuggingFace
        │
        ▼
evaluate.py
  → loads base model + fine-tuned adapters
  → runs both on eval.jsonl
  → calculates ROUGE-1, ROUGE-2, ROUGE-L
  → saves side-by-side qualitative comparisons
        │
        ▼
GitHub (code + results) + HuggingFace (model weights)
```

---

## Infrastructure

| Component | Platform |
|---|---|
| Training compute | Kaggle Notebooks (free T4 GPU, 30hr/week) |
| Code & version control | GitHub (public repo) |
| Model hosting | HuggingFace Hub |
| Local development | Codex / VS Code |

---

## Repo Structure

```
repo/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── inference.py
│   ├── autoresearch_loop.py
│   └── config/
│       └── qlora_config.yaml
├── data/
│   ├── processed/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── eval.jsonl
├── evaluation/
│   ├── evaluate.py
│   └── results/
├── README.md
├── ARCHITECTURE.md
├── EVALUATION.md
├── DATASET.md
├── TRAINING.md
├── DEMO_VIDEO            ← link to Loom/YouTube recording
└── requirements.txt
```