from __future__ import annotations

import argparse
import gc
import inspect
import itertools
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:  # pragma: no cover - fallback for older TRL versions
    SFTConfig = None


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
SEARCH_SPACE = {
    "lora_r": [8, 16, 32],
    "learning_rate": [1e-4, 2e-4, 5e-4],
    "prompt_format": ["alpaca", "chat", "minimal"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short-burst autoresearch for QLoRA Financial QA.")
    parser.add_argument("--train_data", type=Path, required=True, help="Path to train.jsonl.")
    parser.add_argument("--val_data", type=Path, required=True, help="Path to val.jsonl.")
    parser.add_argument(
        "--n_iter",
        type=int,
        default=15,
        help="Number of short-burst experiments to run.",
    )
    parser.add_argument(
        "--output_config",
        type=Path,
        required=True,
        help="Where to save the best config YAML.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model to fine-tune during the experiment search.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for search reproducibility.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code when loading the base model and tokenizer.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU is required for this short-burst QLoRA search.")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompt(instruction: str, response: str | None = None, prompt_format: str = "alpaca") -> str:
    clean_instruction = instruction.strip()
    clean_response = None if response is None else response.strip()

    if prompt_format == "alpaca":
        prompt = f"### Instruction:\n{clean_instruction}\n\n### Response:\n"
    elif prompt_format == "chat":
        prompt = (
            "System: You are a helpful financial assistant.\n"
            f"User: {clean_instruction}\n"
            "Assistant: "
        )
    elif prompt_format == "minimal":
        prompt = f"{clean_instruction}\nAnswer: "
    else:
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

    if clean_response is None:
        return prompt
    return prompt + clean_response


def format_dataset(records: Sequence[Dict[str, Any]], prompt_format: str) -> Dataset:
    formatted = []
    for record in records:
        instruction = record["instruction"]
        response = record.get("response") or record.get("output", "")
        formatted.append(
            {
                **record,
                "text": build_prompt(instruction, response=response, prompt_format=prompt_format),
            }
        )
    return Dataset.from_list(formatted)


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str, trust_remote_code: bool) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_quantized_model(model_name: str, trust_remote_code: bool) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    except TypeError:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    return model


def build_lora_config(lora_r: int) -> LoraConfig:
    return LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_training_args(output_dir: Path, learning_rate: float, seed: int) -> Any:
    common_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": 200,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "fp16": True,
        "optim": "paged_adamw_32bit",
        "logging_steps": 20,
        "save_strategy": "no",
        "report_to": "none",
        "seed": seed,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "disable_tqdm": False,
    }

    target_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    params = set(inspect.signature(target_cls).parameters.keys())

    if "evaluation_strategy" in params:
        common_kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in params:
        common_kwargs["eval_strategy"] = "no"

    if target_cls is SFTConfig:
        if "dataset_text_field" in params:
            common_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in params:
            common_kwargs["max_seq_length"] = 512
        elif "max_length" in params:
            common_kwargs["max_length"] = 512
        if "packing" in params:
            common_kwargs["packing"] = False

    return target_cls(**common_kwargs)


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    training_args: Any,
    peft_config: LoraConfig,
) -> SFTTrainer:
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if SFTConfig is None:
        if "dataset_text_field" in trainer_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in trainer_params:
            trainer_kwargs["max_seq_length"] = 512
        if "packing" in trainer_params:
            trainer_kwargs["packing"] = False

    return SFTTrainer(**trainer_kwargs)


def generate_predictions(
    model: Any,
    tokenizer: Any,
    val_records: Sequence[Dict[str, Any]],
    prompt_format: str,
    max_new_tokens: int = 128,
) -> List[str]:
    predictions: List[str] = []
    model.eval()

    for record in tqdm(val_records, desc="Generating validation predictions", leave=False):
        prompt = build_prompt(record["instruction"], response=None, prompt_format=prompt_format)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_tokens = generated[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
        predictions.append(completion)

    return predictions


def compute_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for prediction, reference in zip(predictions, references):
        scores.append(scorer.score(reference, prediction)["rougeL"].fmeasure)
    return float(sum(scores) / max(1, len(scores)))


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_next_config(
    all_configs: Sequence[Dict[str, Any]],
    tried: Sequence[Dict[str, Any]],
    best_config: Dict[str, Any] | None,
    rng: random.Random,
) -> Dict[str, Any]:
    tried_keys = {config_to_key(config) for config in tried}
    remaining = [config for config in all_configs if config_to_key(config) not in tried_keys]
    if not remaining:
        raise RuntimeError("No remaining configurations left to evaluate.")

    if best_config is None or rng.random() < 0.3:
        return rng.choice(remaining)

    def heuristic(config: Dict[str, Any]) -> float:
        same_dimensions = sum(
            1
            for key in ("lora_r", "learning_rate", "prompt_format")
            if config[key] == best_config[key]
        )
        return same_dimensions + rng.random() * 0.05

    return max(remaining, key=heuristic)


def config_to_key(config: Dict[str, Any]) -> Tuple[Any, ...]:
    return (config["lora_r"], config["learning_rate"], config["prompt_format"])


def run_experiment(
    experiment_index: int,
    config: Dict[str, Any],
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    base_model: str,
    work_root: Path,
    trust_remote_code: bool,
    seed: int,
) -> float:
    experiment_dir = work_root / f"experiment_{experiment_index:02d}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Experiment %d | lora_r=%s | learning_rate=%s | prompt_format=%s",
        experiment_index,
        config["lora_r"],
        config["learning_rate"],
        config["prompt_format"],
    )

    model = tokenizer = trainer = None
    try:
        tokenizer = load_tokenizer(base_model, trust_remote_code=trust_remote_code)
        model = load_quantized_model(base_model, trust_remote_code=trust_remote_code)

        train_dataset = format_dataset(train_records, prompt_format=config["prompt_format"])
        training_args = build_training_args(
            experiment_dir,
            learning_rate=config["learning_rate"],
            seed=seed,
        )
        peft_config = build_lora_config(config["lora_r"])
        trainer = build_trainer(model, tokenizer, train_dataset, training_args, peft_config)

        trainer.train()
        predictions = generate_predictions(
            model=trainer.model,
            tokenizer=tokenizer,
            val_records=val_records,
            prompt_format=config["prompt_format"],
        )
        references = [(record.get("response") or record.get("output", "")).strip() for record in val_records]
        rouge_l = compute_rouge_l(predictions, references)

        with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({"rouge_l": rouge_l, "config": config}, handle, indent=2)

        return rouge_l
    finally:
        del trainer
        del model
        del tokenizer
        cleanup_cuda()


def save_best_config(path: Path, best_config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)


def print_summary_table(results: Sequence[Dict[str, Any]]) -> None:
    headers = ("Iter", "LoRA r", "LR", "Prompt", "ROUGE-L", "Best So Far")
    rows = []
    for row in results:
        rows.append(
            (
                str(row["iteration"]),
                str(row["lora_r"]),
                f"{row['learning_rate']:.0e}",
                row["prompt_format"],
                f"{row['rouge_l']:.4f}",
                "yes" if row["is_best"] else "no",
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(current, len(value)) for current, value in zip(widths, row)]

    separator = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))

    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def main() -> None:
    args = parse_args()
    setup_logging()

    try:
        ensure_cuda_available()
        set_seed(args.seed)

        train_records = read_jsonl(args.train_data.resolve())
        val_records = read_jsonl(args.val_data.resolve())
        logging.info("Loaded %d training records and %d validation records.", len(train_records), len(val_records))

        all_configs = [
            {
                "base_model": args.base_model,
                "lora_r": lora_r,
                "learning_rate": learning_rate,
                "prompt_format": prompt_format,
                "max_steps": 200,
                "metric": "rougeL",
            }
            for lora_r, learning_rate, prompt_format in itertools.product(
                SEARCH_SPACE["lora_r"],
                SEARCH_SPACE["learning_rate"],
                SEARCH_SPACE["prompt_format"],
            )
        ]

        actual_iterations = min(args.n_iter, len(all_configs))
        if actual_iterations < args.n_iter:
            logging.warning(
                "Requested %d iterations, but only %d unique configurations exist. Running %d experiments.",
                args.n_iter,
                len(all_configs),
                actual_iterations,
            )

        rng = random.Random(args.seed)
        work_root = args.output_config.resolve().parent / ".autoresearch_runs"
        work_root.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        best_config: Dict[str, Any] | None = None
        best_score = float("-inf")

        for iteration in tqdm(range(1, actual_iterations + 1), desc="Autoresearch iterations"):
            config = sample_next_config(all_configs, results, best_config, rng)
            is_best = False
            error_message = None

            try:
                score = run_experiment(
                    experiment_index=iteration,
                    config=config,
                    train_records=train_records,
                    val_records=val_records,
                    base_model=args.base_model,
                    work_root=work_root,
                    trust_remote_code=args.trust_remote_code,
                    seed=args.seed,
                )
                is_best = score > best_score
                if is_best:
                    best_score = score
                    best_config = {**config, "rouge_l": score}
                    logging.info("New best config found with ROUGE-L=%.4f", score)
                else:
                    logging.info("Config underperformed current best (%.4f <= %.4f); discarding.", score, best_score)
            except Exception as exc:
                score = float("nan")
                error_message = str(exc)
                logging.exception("Experiment %d failed and will be discarded: %s", iteration, exc)

            results.append(
                {
                    "iteration": iteration,
                    **config,
                    "rouge_l": score,
                    "is_best": is_best,
                    "error": error_message,
                }
            )

            experiment_dir = work_root / f"experiment_{iteration:02d}"
            if experiment_dir.exists() and not is_best:
                shutil.rmtree(experiment_dir, ignore_errors=True)

        if best_config is None:
            raise RuntimeError("No experiments were completed successfully.")

        save_best_config(args.output_config.resolve(), best_config)
        print_summary_table(results)
        logging.info("Best config saved to %s", args.output_config.resolve())
    except Exception as exc:
        logging.exception("Autoresearch loop failed: %s", exc)
        raise SystemExit(1) from exc
    finally:
        cleanup_cuda()


if __name__ == "__main__":
    main()
