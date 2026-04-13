# ARCHITECTURE_DIAGRAM.md

## End-to-End Pipeline

```text
+-------------------------------+
| Hugging Face Dataset          |
| gbharti/finance-alpaca        |
+-------------------------------+
                |
                v
+-------------------------------+
| Data Preparation              |
| - Filter short/long outputs   |
| - Remove HTML                 |
| - Remove near-duplicates      |
| - Balance financial topics    |
+-------------------------------+
                |
                v
+-------------------------------+
| Instruction Formatting        |
| ### Instruction:              |
| {question}                    |
|                               |
| ### Response:                 |
| {answer}                      |
+-------------------------------+
                |
                v
+-------------------------------+
| Dataset Splits                |
| - train.jsonl (80%)           |
| - val.jsonl (10%)             |
| - eval.jsonl (10%)            |
+-------------------------------+


+-------------------------------+
| Base Model                    |
| Qwen/Qwen3-1.7B               |
+-------------------------------+
                |
                v
+-------------------------------+
| 4-bit Quantization            |
| BitsAndBytes NF4              |
| load_in_4bit = True           |
+-------------------------------+
                |
                v
+-------------------------------+
| LoRA Adapters                 |
| r = 16, alpha = 32            |
| q_proj, k_proj, v_proj,       |
| o_proj                        |
+-------------------------------+
                |
                v
+-------------------------------+
| Fine-Tuned Model              |
| Qwen3-1.7B + Financial Q&A    |
| LoRA Adapters                 |
+-------------------------------+


+-------------------------------+
| Autoresearch Loop             |
| Define metric (ROUGE-based)   |
+-------------------------------+
                |
                v
+-------------------------------+
| Short-Burst Training          |
| 200-step experiment           |
+-------------------------------+
                |
                v
+-------------------------------+
| Score Configuration           |
| Validate on val.jsonl         |
+-------------------------------+
                |
                v
+-------------------------------+
| Keep or Discard               |
| Better than best so far?      |
+-------------------------------+
         |                 |
       Yes                 No
         |                 |
         v                 v
+------------------+   +------------------+
| Update Best      |   | Discard Config   |
| Config           |   | and Continue     |
+------------------+   +------------------+
         \               /
          \             /
           v           v
        +-------------------+
        | Repeat Search     |
        | Next Experiment   |
        +-------------------+


+-------------------------------+        +-------------------------------+
| Base Model Inference          |        | Fine-Tuned Model Inference    |
| Generate answers on eval set  |        | Generate answers on eval set  |
+-------------------------------+        +-------------------------------+
                \                              /
                 \                            /
                  v                          v
                 +----------------------------+
                 | Evaluation                 |
                 | - ROUGE-1 / ROUGE-2 / L   |
                 | - BLEU                     |
                 | - Qualitative comparison   |
                 +----------------------------+
                                |
                                v
                 +----------------------------+
                 | Final Results              |
                 | Base vs Fine-Tuned         |
                 +----------------------------+
```
