from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:  # pragma: no cover - fallback for older TRL versions
    SFTConfig = None

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - optional dependency
    HfApi = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-1.7B on Financial QA with QLoRA.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for checkpoints, adapters, and training artifacts.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML config to contain a top-level mapping, found: {type(loaded).__name__}")
    return loaded


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def get_nested(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU is required for 4-bit QLoRA training on this script.")


def build_bnb_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    compute_dtype_name = str(get_nested(config, "quantization.compute_dtype", "float16")).lower()
    compute_dtype = getattr(torch, compute_dtype_name)
    return BitsAndBytesConfig(
        load_in_4bit=bool(get_nested(config, "quantization.load_in_4bit", True)),
        bnb_4bit_quant_type=get_nested(config, "quantization.quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(get_nested(config, "quantization.use_double_quant", True)),
    )


def load_tokenizer(model_name: str, trust_remote_code: bool) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str, bnb_config: BitsAndBytesConfig, trust_remote_code: bool, gradient_checkpointing: bool) -> Any:
    logging.info("Loading base model '%s' in 4-bit mode.", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False

    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    except TypeError:
        model = prepare_model_for_kbit_training(model)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def build_lora_config(config: Dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=int(get_nested(config, "lora.r", 16)),
        lora_alpha=int(get_nested(config, "lora.alpha", 32)),
        lora_dropout=float(get_nested(config, "lora.dropout", 0.05)),
        target_modules=list(
            get_nested(
                config,
                "lora.target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            )
        ),
        bias="none",
        task_type=TaskType[get_nested(config, "lora.task_type", "CAUSAL_LM")],
    )


def build_training_args(config: Dict[str, Any], output_dir: Path) -> Any:
    common_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": float(get_nested(config, "training.epochs", 3)),
        "per_device_train_batch_size": int(get_nested(config, "training.batch_size", 4)),
        "per_device_eval_batch_size": int(get_nested(config, "training.eval_batch_size", 4)),
        "gradient_accumulation_steps": int(get_nested(config, "training.gradient_accumulation_steps", 4)),
        "learning_rate": float(get_nested(config, "training.learning_rate", 2e-4)),
        "lr_scheduler_type": get_nested(config, "training.lr_scheduler", "cosine"),
        "warmup_ratio": float(get_nested(config, "training.warmup_ratio", 0.03)),
        "fp16": bool(get_nested(config, "training.fp16", True)),
        "bf16": bool(get_nested(config, "training.bf16", False)),
        "optim": get_nested(config, "training.optim", "paged_adamw_32bit"),
        "logging_steps": int(get_nested(config, "training.logging_steps", 10)),
        "save_strategy": get_nested(config, "training.save_strategy", "epoch"),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": int(get_nested(config, "training.save_total_limit", 1)),
        "report_to": get_nested(config, "training.report_to", "none"),
        "seed": int(get_nested(config, "training.seed", 42)),
        "gradient_checkpointing": bool(get_nested(config, "training.gradient_checkpointing", True)),
        "remove_unused_columns": False,
        "max_grad_norm": float(get_nested(config, "training.max_grad_norm", 1.0)),
        "weight_decay": float(get_nested(config, "training.weight_decay", 0.0)),
        "dataloader_pin_memory": bool(get_nested(config, "training.dataloader_pin_memory", False)),
    }

    target_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    params = set(inspect.signature(target_cls).parameters.keys())

    if "evaluation_strategy" in params:
        common_kwargs["evaluation_strategy"] = get_nested(config, "training.eval_strategy", "epoch")
    elif "eval_strategy" in params:
        common_kwargs["eval_strategy"] = get_nested(config, "training.eval_strategy", "epoch")

    if target_cls is SFTConfig:
        if "dataset_text_field" in params:
            common_kwargs["dataset_text_field"] = get_nested(config, "data.text_field", "text")
        if "max_seq_length" in params:
            common_kwargs["max_seq_length"] = int(get_nested(config, "training.max_seq_length", 512))
        elif "max_length" in params:
            common_kwargs["max_length"] = int(get_nested(config, "training.max_seq_length", 512))
        if "packing" in params:
            common_kwargs["packing"] = bool(get_nested(config, "training.packing", False))

    return target_cls(**common_kwargs)


def load_datasets(config: Dict[str, Any], config_base_dir: Path) -> Dict[str, Any]:
    train_path = resolve_path(config_base_dir, get_nested(config, "data.train_file", "data/train.jsonl"))
    val_path = resolve_path(config_base_dir, get_nested(config, "data.val_file", "data/val.jsonl"))

    logging.info("Loading training data from %s", train_path)
    logging.info("Loading validation data from %s", val_path)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(val_path),
        },
    )
    return dataset


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    training_args: Any,
    peft_config: LoraConfig,
    config: Dict[str, Any],
) -> SFTTrainer:
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if SFTConfig is None:
        if "dataset_text_field" in trainer_params:
            trainer_kwargs["dataset_text_field"] = get_nested(config, "data.text_field", "text")
        if "max_seq_length" in trainer_params:
            trainer_kwargs["max_seq_length"] = int(get_nested(config, "training.max_seq_length", 512))
        if "packing" in trainer_params:
            trainer_kwargs["packing"] = bool(get_nested(config, "training.packing", False))

    return SFTTrainer(**trainer_kwargs)


def save_training_artifacts(
    trainer: SFTTrainer,
    tokenizer: Any,
    output_dir: Path,
) -> Path:
    adapters_dir = output_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Saving LoRA adapters to %s", adapters_dir)
    trainer.model.save_pretrained(adapters_dir)
    tokenizer.save_pretrained(adapters_dir)

    training_state_path = output_dir / "training_metrics.json"
    training_summary = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
    }
    with training_state_path.open("w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2)

    return adapters_dir


def resolve_hub_repo_id(config: Dict[str, Any], output_dir: Path, token: str) -> str:
    configured_repo = get_nested(config, "hub.repo_id")
    if configured_repo:
        return str(configured_repo)

    username = os.getenv("HF_USERNAME") or os.getenv("HUGGINGFACE_USERNAME")
    if not username and HfApi is not None:
        try:
            username = HfApi().whoami(token=token)["name"]
        except Exception as exc:  # pragma: no cover - depends on external auth state
            logging.warning("Unable to infer Hugging Face username automatically: %s", exc)

    if not username:
        raise ValueError(
            "HF_TOKEN is set, but no hub repo id or username is available. "
            "Set hub.repo_id in the config or export HF_USERNAME."
        )

    repo_name = get_nested(config, "hub.repo_name", output_dir.name)
    return f"{username}/{repo_name}"


def maybe_push_to_hub(config: Dict[str, Any], model: Any, tokenizer: Any, output_dir: Path) -> None:
    token = os.getenv("HF_TOKEN")
    should_push = bool(get_nested(config, "hub.push_if_token_available", True))

    if not token or not should_push:
        logging.info("HF_TOKEN not set or push disabled; skipping Hugging Face Hub upload.")
        return

    repo_id = resolve_hub_repo_id(config, output_dir, token)
    private = bool(get_nested(config, "hub.private", False))
    logging.info("Pushing adapters to Hugging Face Hub repo: %s", repo_id)

    model.push_to_hub(repo_id, token=token, private=private)
    tokenizer.push_to_hub(repo_id, token=token, private=private)
    logging.info("Hub push completed successfully.")


def main() -> None:
    args = parse_args()
    setup_logging()

    try:
        ensure_cuda_available()

        config = load_yaml_config(args.config)
        config_base_dir = args.config.parent.resolve()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        seed = int(get_nested(config, "training.seed", 42))
        set_seed(seed)
        logging.info("Using random seed: %d", seed)

        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        model_name = get_nested(config, "model.name", "Qwen/Qwen3-1.7B")
        trust_remote_code = bool(get_nested(config, "model.trust_remote_code", False))
        gradient_checkpointing = bool(get_nested(config, "training.gradient_checkpointing", True))

        dataset = load_datasets(config, config_base_dir)
        tokenizer = load_tokenizer(model_name, trust_remote_code=trust_remote_code)
        bnb_config = build_bnb_config(config)
        model = load_model(
            model_name=model_name,
            bnb_config=bnb_config,
            trust_remote_code=trust_remote_code,
            gradient_checkpointing=gradient_checkpointing,
        )
        peft_config = build_lora_config(config)
        training_args = build_training_args(config, output_dir)
        trainer = build_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            training_args=training_args,
            peft_config=peft_config,
            config=config,
        )

        logging.info("Starting supervised fine-tuning.")
        train_result = trainer.train()
        logging.info("Training finished. Global step: %s", trainer.state.global_step)

        metrics_path = output_dir / "trainer_train_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(train_result.metrics, handle, indent=2)

        eval_metrics = trainer.evaluate()
        with (output_dir / "trainer_eval_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(eval_metrics, handle, indent=2)

        trainer.model.config.use_cache = True
        adapters_dir = save_training_artifacts(trainer, tokenizer, output_dir)
        maybe_push_to_hub(config, trainer.model, tokenizer, output_dir)

        logging.info("Best checkpoint: %s", trainer.state.best_model_checkpoint)
        logging.info("Best eval_loss: %s", trainer.state.best_metric)
        logging.info("Training pipeline completed successfully.")
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        raise SystemExit(1) from exc
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
