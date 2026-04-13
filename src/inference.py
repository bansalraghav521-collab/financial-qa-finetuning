from __future__ import annotations

import argparse
import gc
import logging
import shutil
import textwrap
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned Financial QA generations.")
    parser.add_argument("--adapter_path", type=Path, required=True, help="Path to the saved LoRA adapters.")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or local model path.",
    )
    parser.add_argument("--question", type=str, required=True, help="Financial question to ask both models.")
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Maximum number of tokens to generate.")
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
        raise EnvironmentError("CUDA GPU is required for 4-bit inference with this script.")


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


def load_base_model(model_name: str, trust_remote_code: bool) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


def load_finetuned_model(model_name: str, adapter_path: Path, trust_remote_code: bool) -> Any:
    base_model = load_base_model(model_name, trust_remote_code=trust_remote_code)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    return model


def build_prompt(question: str) -> str:
    return f"### Instruction:\n{question.strip()}\n\n### Response:\n"


def generate_response(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = generated[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def wrap_lines(text: str, width: int) -> list[str]:
    wrapped = textwrap.wrap(text, width=width) if text else [""]
    return wrapped or [""]


def print_side_by_side(left_title: str, left_text: str, right_title: str, right_text: str) -> None:
    terminal_width = shutil.get_terminal_size(fallback=(140, 40)).columns
    padding = 4
    column_width = max(40, (terminal_width - padding) // 2)
    separator = " " * padding

    left_lines = [left_title, "-" * column_width, *wrap_lines(left_text, column_width)]
    right_lines = [right_title, "-" * column_width, *wrap_lines(right_text, column_width)]
    max_lines = max(len(left_lines), len(right_lines))

    left_lines.extend([""] * (max_lines - len(left_lines)))
    right_lines.extend([""] * (max_lines - len(right_lines)))

    for left, right in zip(left_lines, right_lines):
        print(f"{left.ljust(column_width)}{separator}{right.ljust(column_width)}")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    setup_logging()

    base_model = None
    finetuned_model = None

    try:
        ensure_cuda_available()
        adapter_path = args.adapter_path.resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        prompt = build_prompt(args.question)
        tokenizer = load_tokenizer(args.base_model, trust_remote_code=args.trust_remote_code)

        logging.info("Running inference with the base model.")
        base_model = load_base_model(args.base_model, trust_remote_code=args.trust_remote_code)
        base_output = generate_response(base_model, tokenizer, prompt, args.max_new_tokens)
        del base_model
        base_model = None
        cleanup_cuda()

        logging.info("Running inference with the fine-tuned adapter model.")
        finetuned_model = load_finetuned_model(
            args.base_model,
            adapter_path,
            trust_remote_code=args.trust_remote_code,
        )
        finetuned_output = generate_response(finetuned_model, tokenizer, prompt, args.max_new_tokens)

        print_side_by_side("Base Model Output", base_output, "Fine-Tuned Model Output", finetuned_output)
    except Exception as exc:
        logging.exception("Inference failed: %s", exc)
        raise SystemExit(1) from exc
    finally:
        del base_model
        del finetuned_model
        cleanup_cuda()


if __name__ == "__main__":
    main()
