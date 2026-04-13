from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import sacrebleu
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned Financial QA models.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or local path.")
    parser.add_argument("--adapter_path", type=Path, required=True, help="Path to LoRA adapters.")
    parser.add_argument("--eval_data", type=Path, required=True, help="Path to eval.jsonl.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for saved predictions and score files.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Maximum number of generation tokens.")
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
        raise EnvironmentError("CUDA GPU is required for this evaluation workflow.")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompt(question: str) -> str:
    return f"### Instruction:\n{question.strip()}\n\n### Response:\n"


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


def generate_outputs(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    max_new_tokens: int,
    description: str,
) -> List[str]:
    outputs: List[str] = []
    for record in tqdm(records, desc=description):
        prompt = build_prompt(record["instruction"])
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
        outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return outputs


def compute_scores(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for prediction, reference in zip(predictions, references):
        score = scorer.score(reference, prediction)
        for metric in rouge_totals:
            rouge_totals[metric] += score[metric].fmeasure

    count = max(1, len(predictions))
    averaged = {metric: total / count for metric, total in rouge_totals.items()}
    averaged["bleu"] = float(sacrebleu.corpus_bleu(list(predictions), [list(references)]).score)
    return averaged


def save_outputs(path: Path, records: Sequence[Dict[str, Any]], predictions: Sequence[str]) -> None:
    payload = []
    for record, prediction in zip(records, predictions):
        payload.append(
            {
                "id": record.get("id"),
                "topic": record.get("topic"),
                "instruction": record["instruction"],
                "reference": record.get("response") or record.get("output", ""),
                "prediction": prediction,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_scores(path: Path, base_scores: Dict[str, float], finetuned_scores: Dict[str, float]) -> None:
    delta = {
        metric: finetuned_scores[metric] - base_scores[metric]
        for metric in base_scores
    }
    payload = {
        "base_model": base_scores,
        "fine_tuned_model": finetuned_scores,
        "delta": delta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def print_summary_table(base_scores: Dict[str, float], finetuned_scores: Dict[str, float]) -> None:
    headers = ("Metric", "Base", "Fine-Tuned", "Delta")
    metrics = ["rouge1", "rouge2", "rougeL", "bleu"]
    rows = []
    for metric in metrics:
        base_value = base_scores[metric]
        tuned_value = finetuned_scores[metric]
        rows.append(
            (
                metric.upper(),
                f"{base_value:.4f}",
                f"{tuned_value:.4f}",
                f"{(tuned_value - base_value):+.4f}",
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
        records = read_jsonl(args.eval_data.resolve())
        references = [(record.get("response") or record.get("output", "")).strip() for record in records]
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Loaded %d evaluation samples.", len(records))
        tokenizer = load_tokenizer(args.base_model, trust_remote_code=args.trust_remote_code)

        logging.info("Generating outputs from the base model.")
        base_model = load_base_model(args.base_model, trust_remote_code=args.trust_remote_code)
        base_predictions = generate_outputs(
            model=base_model,
            tokenizer=tokenizer,
            records=records,
            max_new_tokens=args.max_new_tokens,
            description="Base model evaluation",
        )
        save_outputs(output_dir / "base_model_outputs.json", records, base_predictions)
        base_scores = compute_scores(base_predictions, references)
        del base_model
        base_model = None
        cleanup_cuda()

        logging.info("Generating outputs from the fine-tuned adapter model.")
        finetuned_model = load_finetuned_model(
            args.base_model,
            args.adapter_path.resolve(),
            trust_remote_code=args.trust_remote_code,
        )
        finetuned_predictions = generate_outputs(
            model=finetuned_model,
            tokenizer=tokenizer,
            records=records,
            max_new_tokens=args.max_new_tokens,
            description="Fine-tuned model evaluation",
        )
        save_outputs(output_dir / "finetuned_outputs.json", records, finetuned_predictions)
        finetuned_scores = compute_scores(finetuned_predictions, references)

        save_scores(output_dir / "rouge_scores.json", base_scores, finetuned_scores)
        print_summary_table(base_scores, finetuned_scores)
        logging.info("Evaluation completed successfully. Results saved to %s", output_dir)
    except Exception as exc:
        logging.exception("Evaluation failed: %s", exc)
        raise SystemExit(1) from exc
    finally:
        del base_model
        del finetuned_model
        cleanup_cuda()


if __name__ == "__main__":
    main()
