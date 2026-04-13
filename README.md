# Financial Q&A — Fine-Tuning Qwen3-1.7B with QLoRA

Fine-tuned Qwen3-1.7B for accurate Financial Question Answering using QLoRA. Demonstrates measurable improvement over the base model across ROUGE, BLEU, and qualitative metrics. Includes an autonomous hyperparameter search loop inspired by Karpathy's autoresearch concept.

## Why Qwen3-1.7B?
- Small enough to fine-tune on a free Kaggle T4 GPU (16GB VRAM)
- Strong instruction-following baseline out of the box
- Supports 4-bit QLoRA without accuracy collapse
- Trade-off: lower ceiling on complex multi-step reasoning vs larger models, but acceptable for Financial Q&A where answers are factual and concise

## Results

| Metric | Base Model | Fine-tuned | Improvement |
|---|---|---|---|
| ROUGE-1 | 0.1808 | 0.2813 | +55% |
| ROUGE-2 | 0.0874 | 0.1417 | +62% |
| ROUGE-L | 0.1687 | 0.2336 | +38% |
| BLEU | 1.1640 | 2.6316 | +126% |
| Qualitative avg (1-5) | 3.8 | 4.2 | +11% |

## Model
- Base: Qwen/Qwen3-1.7B
- Method: QLoRA (4-bit NF4 + LoRA adapters)
- LoRA rank: 16, alpha: 32, target modules: q_proj, k_proj, v_proj, o_proj
- HuggingFace: https://huggingface.co/IItianRaghav/qwen3-finance-qlora

## Dataset
- Source: gbharti/finance-alpaca (HuggingFace)
- 1,500 curated samples from 68,000 total
- Balanced across 7 topics: interest rates, stocks, personal finance, economic indicators, financial instruments, tax, risk/portfolio
- Cleaned: removed HTML, duplicates, short/long outliers
- Formatted in Alpaca instruction-tuning style
- Split: 80% train / 10% val / 10% eval
- Sample data: data/sample_data.jsonl

## Training Pipeline
- Tokenization: AutoTokenizer with right-padding, EOS as pad token
- Training: SFTTrainer (HuggingFace TRL) with QLoRA
- Hyperparameters: lr=2e-4, epochs=3, batch_size=4, grad_accumulation=4, scheduler=cosine, warmup_ratio=0.03
- Optimizer: paged_adamw_32bit
- Platform: Kaggle T4 GPU (16GB VRAM)
- Training time: ~70 minutes

## Autoresearch Loop
Inspired by Karpathy's autoresearch concept. Instead of manually guessing hyperparameters, an autonomous loop ran 6 experiments over LoRA rank (8, 16, 32) and learning rate (1e-4, 2e-4, 5e-4) combinations. Each experiment trained for 200 steps and was scored using combined ROUGE metric. Result: confirmed original config (r=16, lr=2e-4) as optimal.

## Bonus
- 4-bit quantization via bitsandbytes (QLoRA)
- Autoresearch hyperparameter optimization loop

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with 12GB+ VRAM (tested on Kaggle T4)

### Install
```bash
pip install -r requirements.txt
```

### Run

**1. Prepare dataset:**
```bash
python src/prepare_data.py --output_dir ./data/ --seed 42
```

**2. Train:**
```bash
python src/train.py --config src/config/qlora_config.yaml --output_dir ./outputs
```

**3. Evaluate:**
```bash
python evaluation/evaluate.py \
  --base_model Qwen/Qwen3-1.7B \
  --adapter_path ./outputs/adapters \
  --eval_data ./data/eval.jsonl \
  --output_dir ./evaluation/results/
```

**4. Inference:**
```bash
python src/inference.py --adapter_path ./outputs/adapters --question "What is compound interest?"
```

## Repo Structure
repo/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── inference.py
│   ├── autoresearch_loop.py
│   └── config/
│       └── qlora_config.yaml
├── data/
│   └── sample_data.jsonl
├── evaluation/
│   └── evaluate.py
├── README.md
├── ARCHITECTURE.md
├── ARCHITECTURE_DIAGRAM.md
├── EVALUATION.md
├── DATASET.md
├── TRAINING.md
├── DEMO_VIDEO
└── requirements.txt

## Documentation
| File | Contents |
|---|---|
| ARCHITECTURE.md | Full system design and data flow |
| EVALUATION.md | Complete evaluation results with real scores |
| DATASET.md | Dataset source, cleaning steps, format |
| TRAINING.md | Hyperparameters, LoRA config, training details |
