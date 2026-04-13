# TRAINING.md — Training Details & Configuration

## Approach

**Method:** QLoRA (Quantized Low-Rank Adaptation)  
**Why QLoRA over full fine-tuning:**

| Factor | Full Fine-Tuning | QLoRA |
|---|---|---|
| GPU VRAM needed | 40–80GB | 10–16GB |
| Training time | Days | Hours |
| Cost | High | Free (Kaggle T4) |
| Performance gap | Marginal improvement | Comparable for domain adaptation |
| Adapter size | Full model (~3GB) | ~20–80MB |

For domain adaptation of an SLM on a specific task, QLoRA achieves near-equivalent results to full fine-tuning at a fraction of the compute cost. This is the standard approach in production fine-tuning pipelines.

---

## Model

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen3-1.7B` |
| Quantization | 4-bit NF4 (`bitsandbytes`) |
| Compute dtype | `float16` |
| Double quantization | Enabled |

**Why Qwen3 Mini (1.7B)?**
- Fits comfortably in Kaggle T4 GPU (16GB VRAM) with 4-bit quantization
- Strong base instruction-following capability — responds well to financial prompts even before fine-tuning
- Smaller than Mistral 7B, which means faster iteration during the autoresearch loop
- Trade-off: lower ceiling on complex multi-step reasoning vs Mistral 7B, but acceptable for Financial Q&A where answers are factual and concise

---

## LoRA Configuration

```yaml
lora_r: 16               # Rank — controls adapter size vs expressiveness
lora_alpha: 32           # Scaling: alpha/r = 2.0 (standard rule of thumb)
lora_dropout: 0.05       # Light dropout to reduce overfitting on small dataset
bias: none
task_type: CAUSAL_LM

target_modules:
  - q_proj               # Query projection in attention
  - k_proj               # Key projection
  - v_proj               # Value projection
  - o_proj               # Output projection
```

**Why these modules?** Targeting all four attention projections gives the adapter maximum ability to shift how the model attends to financial vocabulary and concepts, without touching the MLP layers (which would increase adapter size significantly).

**LoRA rank justification:**
- `r=8`: Fewer parameters, faster, but may underfit financial domain nuances
- `r=16`: Chosen — good balance of expressiveness and training stability
- `r=32`: More expressive, but higher risk of overfitting on 1,200 samples

The autoresearch loop validates this choice empirically by testing all three.

---

## Training Hyperparameters

```yaml
# Trainer settings
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4    # Effective batch size = 16
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.001
max_grad_norm: 0.3
fp16: true
optim: paged_adamw_32bit         # Memory-efficient optimizer

# Sequence
max_seq_length: 512
packing: false                   # Disabled — samples have varying lengths

# Logging
logging_steps: 10
eval_strategy: epoch
save_strategy: epoch
load_best_model_at_end: true
metric_for_best_model: eval_loss
```

**Learning rate rationale:** `2e-4` is the standard starting point for LoRA fine-tuning. Too high (e.g., `5e-4`) causes loss instability on small datasets; too low (`5e-5`) causes under-adaptation in 3 epochs. The autoresearch loop validates this.

**Epochs:** 3 epochs gives the model enough passes to adapt to financial vocabulary without overfitting the 1,200-sample training set. Val loss is monitored per epoch and best checkpoint is saved.

---

## Autoresearch Experiment Loop

Before the final full training run, an autonomous short-burst experiment loop finds the best configuration.

**Loop parameters:**
- Iterations: 12–18 experiments
- Duration per experiment: ~300 training steps (~5 minutes on T4)
- Metric: ROUGE-L on `val.jsonl`
- Search space:

```python
search_space = {
    "lora_r": [8, 16, 32],
    "learning_rate": [1e-4, 2e-4, 5e-4],
    "prompt_format": ["alpaca", "chat", "minimal"],
}
```

**Loop output:** A `best_config.yaml` file containing the winning combination, which is then passed directly to the full training script.

```bash
# Run experiment loop
python src/autoresearch_loop.py \
  --train_data ./data/train.jsonl \
  --val_data ./data/val.jsonl \
  --output_config ./src/config/best_config.yaml \
  --n_iter 15

# Full training with best config
python src/train.py --config ./src/config/best_config.yaml
```

---

## Training Environment

| Component | Spec |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA T4 (16GB VRAM) |
| CUDA | 12.x |
| Python | 3.10 |
| OS | Ubuntu 20.04 |
| Training time (estimated) | ~2–3 hours (full run) |
| Autoresearch loop time | ~1.5–2 hours |

---

## Checkpointing & Saving

- Checkpoints saved at end of each epoch
- Best checkpoint (lowest `eval_loss`) selected automatically via `load_best_model_at_end: true`
- LoRA adapters exported separately using `model.save_pretrained()`
- Uploaded to HuggingFace Hub at end of training run

```python
# Save adapters
model.save_pretrained("./adapters/")
tokenizer.save_pretrained("./adapters/")

# Push to HuggingFace
model.push_to_hub("username/qwen3-finance-qlora")
tokenizer.push_to_hub("username/qwen3-finance-qlora")
```

---

## Training Curves

Training and validation loss curves will be logged via HuggingFace Trainer's default logging and saved as plots in `evaluation/results/training_curves.png` after the training run completes.

Expected behavior:
- Training loss: steadily decreasing across 3 epochs
- Val loss: decreasing then plateauing — if it starts rising, the best checkpoint from epoch 2 is used (early stopping)

---

## Bonus — Post-Training Quantization

After fine-tuning, the merged model (base + adapters) is quantized to 4-bit for efficient inference:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

This reduces the final model size further and speeds up inference — a clean bonus for the submission report.

---

## Reproducing Training

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset
python src/prepare_data.py --output_dir ./data/

# 3. Run autoresearch loop (optional but recommended)
python src/autoresearch_loop.py --n_iter 15

# 4. Full training run
python src/train.py --config ./src/config/best_config.yaml

# 5. Evaluate
python evaluation/evaluate.py --adapter_path ./adapters/
```

Full training requires a GPU with at least 12GB VRAM. Tested on Kaggle T4 (16GB).