from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from datasets import load_dataset
from tqdm.auto import tqdm


TOPIC_KEYWORDS: Dict[str, Sequence[str]] = {
    "interest_rates": (
        "interest rate",
        "fed",
        "federal reserve",
        "apr",
        "apy",
        "compound interest",
        "mortgage rate",
        "borrowing cost",
        "yield curve",
        "benchmark rate",
        "rate hike",
        "rate cut",
    ),
    "stocks": (
        "stock",
        "stocks",
        "equity",
        "equities",
        "share price",
        "dividend",
        "market cap",
        "earnings per share",
        "eps",
        "ipo",
        "bull market",
        "bear market",
        "nasdaq",
        "s&p",
    ),
    "personal_finance": (
        "budget",
        "saving",
        "savings",
        "debt",
        "credit score",
        "emergency fund",
        "retirement",
        "401k",
        "ira",
        "personal loan",
        "financial planning",
        "household finance",
        "student loan",
    ),
    "economic_indicators": (
        "gdp",
        "inflation",
        "cpi",
        "ppi",
        "unemployment",
        "economic indicator",
        "recession",
        "consumer confidence",
        "pmi",
        "monetary policy",
        "fiscal policy",
        "trade deficit",
        "economic growth",
    ),
    "financial_instruments": (
        "bond",
        "bonds",
        "etf",
        "mutual fund",
        "option",
        "options",
        "futures",
        "swap",
        "derivative",
        "derivatives",
        "security",
        "securities",
        "treasury bill",
        "certificate of deposit",
        "commercial paper",
    ),
    "tax": (
        "tax",
        "taxes",
        "deduction",
        "deductions",
        "capital gains",
        "withholding",
        "irs",
        "taxable",
        "tax bracket",
        "tax return",
        "vat",
        "tax liability",
    ),
    "risk_portfolio": (
        "risk",
        "portfolio",
        "diversification",
        "asset allocation",
        "volatility",
        "beta",
        "hedge",
        "sharpe",
        "correlation",
        "rebalancing",
        "drawdown",
        "risk-adjusted",
        "expected return",
    ),
}

HTML_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"\b\w+\b")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")


@dataclass
class PreparedSample:
    sample_id: str
    instruction: str
    response: str
    matched_topics: List[str]
    topic_scores: Dict[str, int]
    text: str
    topic: str | None = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "id": self.sample_id,
            "instruction": self.instruction,
            "response": self.response,
            "output": self.response,
            "topic": self.topic,
            "matched_topics": self.matched_topics,
            "text": self.text,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a balanced Finance Alpaca subset.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Directory where train.jsonl, val.jsonl, and eval.jsonl will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling and sampling.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1500,
        help="Total number of curated samples to keep across all financial topics.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def contains_html(text: str) -> bool:
    return bool(HTML_PATTERN.search(text))


def get_first_available(record: Dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def combine_instruction(record: Dict[str, Any]) -> str:
    instruction = get_first_available(record, ("instruction", "question", "prompt"))
    extra_context = get_first_available(record, ("input", "context"))
    instruction = normalize_text(instruction)
    extra_context = normalize_text(extra_context)

    if extra_context:
        return f"{instruction}\n\nContext: {extra_context}"
    return instruction


def extract_response(record: Dict[str, Any]) -> str:
    response = get_first_available(record, ("output", "response", "answer", "completion"))
    return normalize_text(response)


def normalize_instruction_for_dedupe(text: str) -> str:
    lowered = text.lower()
    lowered = NON_ALNUM_PATTERN.sub(" ", lowered)
    return normalize_text(lowered)


def build_bucket_keys(normalized_instruction: str) -> List[str]:
    tokens = normalized_instruction.split()
    prefix = normalized_instruction[:32]
    first_words = " ".join(tokens[:4])
    last_words = " ".join(tokens[-4:])
    sorted_words = " ".join(sorted(set(tokens))[:6])
    length_bucket = str(len(normalized_instruction) // 20)

    keys = {
        f"prefix:{prefix}",
        f"first:{first_words}",
        f"last:{last_words}",
        f"len:{length_bucket}:{sorted_words}",
    }
    return [key for key in keys if key and not key.endswith(":")]


def score_topics(text: str) -> Dict[str, int]:
    lowered = text.lower()
    scores: Dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        scores[topic] = score
    return scores


def matched_topics_from_scores(scores: Dict[str, int]) -> List[str]:
    ranked = [topic for topic, score in scores.items() if score > 0]
    ranked.sort(key=lambda topic: (-scores[topic], topic))
    return ranked


def format_alpaca_sample(instruction: str, response: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def filter_dataset(raw_records: Sequence[Dict[str, Any]]) -> List[PreparedSample]:
    filtered: List[PreparedSample] = []

    for index, record in enumerate(tqdm(raw_records, desc="Filtering raw samples")):
        instruction = combine_instruction(record)
        response = extract_response(record)

        if len(instruction) <= 10:
            continue

        response_word_count = count_words(response)
        if response_word_count < 20 or response_word_count > 300:
            continue

        if contains_html(instruction) or contains_html(response):
            continue

        topic_scores = score_topics(f"{instruction}\n{response}")
        matched_topics = matched_topics_from_scores(topic_scores)

        sample = PreparedSample(
            sample_id=f"finance-alpaca-{index}",
            instruction=instruction,
            response=response,
            matched_topics=matched_topics,
            topic_scores=topic_scores,
            text=format_alpaca_sample(instruction, response),
        )
        filtered.append(sample)

    return filtered


def remove_near_duplicates(samples: Sequence[PreparedSample], threshold: float = 0.85) -> List[PreparedSample]:
    kept_samples: List[PreparedSample] = []
    kept_normalized: List[str] = []
    buckets: Dict[str, List[int]] = defaultdict(list)

    for sample in tqdm(samples, desc="Removing near-duplicate instructions"):
        normalized_instruction = normalize_instruction_for_dedupe(sample.instruction)
        candidate_indices = set()

        for key in build_bucket_keys(normalized_instruction):
            candidate_indices.update(buckets.get(key, []))

        is_duplicate = False
        for candidate_index in candidate_indices:
            existing = kept_normalized[candidate_index]
            length_gap = abs(len(normalized_instruction) - len(existing))
            length_limit = max(10, int(max(len(normalized_instruction), len(existing)) * 0.25))
            if length_gap > length_limit:
                continue

            similarity = SequenceMatcher(None, normalized_instruction, existing).ratio()
            if similarity > threshold:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        kept_index = len(kept_samples)
        kept_samples.append(sample)
        kept_normalized.append(normalized_instruction)
        for key in build_bucket_keys(normalized_instruction):
            buckets[key].append(kept_index)

    return kept_samples


def topic_quotas(total_samples: int) -> Dict[str, int]:
    topics = list(TOPIC_KEYWORDS.keys())
    base_quota = total_samples // len(topics)
    remainder = total_samples % len(topics)
    quotas = {topic: base_quota for topic in topics}
    for topic in topics[:remainder]:
        quotas[topic] += 1
    return quotas


def select_balanced_samples(samples: Sequence[PreparedSample], n_samples: int, seed: int) -> List[PreparedSample]:
    rng = random.Random(seed)
    eligible = [sample for sample in samples if sample.matched_topics]
    if len(eligible) < n_samples:
        raise ValueError(
            f"Only {len(eligible)} topic-matched samples remain after filtering, fewer than requested {n_samples}."
        )

    shuffled = eligible[:]
    rng.shuffle(shuffled)

    pool_sizes = {
        topic: sum(1 for sample in shuffled if topic in sample.matched_topics)
        for topic in TOPIC_KEYWORDS
    }
    quotas = topic_quotas(n_samples)
    selected: List[PreparedSample] = []
    used_ids = set()
    per_topic_selected = defaultdict(int)

    topic_order = sorted(TOPIC_KEYWORDS.keys(), key=lambda topic: (pool_sizes[topic], topic))

    for topic in topic_order:
        needed = quotas[topic]
        candidates = [
            sample
            for sample in shuffled
            if sample.sample_id not in used_ids and topic in sample.matched_topics
        ]
        candidates.sort(
            key=lambda sample: (
                len(sample.matched_topics),
                -sample.topic_scores.get(topic, 0),
                sample.sample_id,
            )
        )

        for sample in candidates[:needed]:
            selected_sample = PreparedSample(
                sample_id=sample.sample_id,
                instruction=sample.instruction,
                response=sample.response,
                matched_topics=list(sample.matched_topics),
                topic_scores=dict(sample.topic_scores),
                text=sample.text,
                topic=topic,
            )
            selected.append(selected_sample)
            used_ids.add(sample.sample_id)
            per_topic_selected[topic] += 1

    if len(selected) < n_samples:
        logging.warning(
            "Perfect topic balancing was not possible with the available pools. Filling the remainder from the best available unused samples."
        )
        remaining = [sample for sample in shuffled if sample.sample_id not in used_ids]
        remaining.sort(
            key=lambda sample: (
                min(per_topic_selected.get(topic, 0) for topic in sample.matched_topics) if sample.matched_topics else 10**6,
                len(sample.matched_topics),
                sample.sample_id,
            )
        )
        for sample in remaining:
            if len(selected) >= n_samples:
                break

            if sample.matched_topics:
                assigned_topic = min(
                    sample.matched_topics,
                    key=lambda topic: (per_topic_selected.get(topic, 0), topic),
                )
            else:
                assigned_topic = "uncategorized"

            selected_sample = PreparedSample(
                sample_id=sample.sample_id,
                instruction=sample.instruction,
                response=sample.response,
                matched_topics=list(sample.matched_topics),
                topic_scores=dict(sample.topic_scores),
                text=sample.text,
                topic=assigned_topic,
            )
            selected.append(selected_sample)
            used_ids.add(sample.sample_id)
            per_topic_selected[assigned_topic] += 1

    if len(selected) < n_samples:
        raise RuntimeError(
            f"Unable to assemble {n_samples} samples. Only {len(selected)} could be selected after balancing."
        )

    rng.shuffle(selected)
    return selected[:n_samples]


def stratified_split(samples: Sequence[PreparedSample], seed: int) -> Dict[str, List[PreparedSample]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[PreparedSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.topic or "uncategorized"].append(sample)

    splits = {"train": [], "val": [], "eval": []}

    for topic, group in grouped.items():
        rng.shuffle(group)
        total = len(group)
        train_count = int(total * 0.8)
        val_count = int(total * 0.1)
        eval_count = total - train_count - val_count

        if total >= 10 and val_count == 0:
            val_count = 1
            train_count -= 1
        if total >= 10 and eval_count == 0:
            eval_count = 1
            train_count -= 1

        splits["train"].extend(group[:train_count])
        splits["val"].extend(group[train_count : train_count + val_count])
        splits["eval"].extend(group[train_count + val_count :])
        logging.info(
            "Topic '%s' split into %d train / %d val / %d eval samples.",
            topic,
            train_count,
            val_count,
            eval_count,
        )

    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits


def write_jsonl(path: Path, samples: Sequence[PreparedSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")


def log_split_summary(splits: Dict[str, List[PreparedSample]]) -> None:
    for split_name, split_samples in splits.items():
        topic_counts = defaultdict(int)
        for sample in split_samples:
            topic_counts[sample.topic or "uncategorized"] += 1
        logging.info("Split '%s' contains %d samples.", split_name, len(split_samples))
        logging.info("Topic distribution for '%s': %s", split_name, dict(sorted(topic_counts.items())))


def main() -> None:
    args = parse_args()
    setup_logging()

    try:
        logging.info("Downloading dataset: gbharti/finance-alpaca")
        dataset = load_dataset("gbharti/finance-alpaca", split="train")
        raw_records = [dict(record) for record in dataset]
        logging.info("Loaded %d raw records.", len(raw_records))

        filtered_samples = filter_dataset(raw_records)
        logging.info("Retained %d samples after basic filtering.", len(filtered_samples))

        deduplicated_samples = remove_near_duplicates(filtered_samples, threshold=0.85)
        logging.info("Retained %d samples after near-duplicate removal.", len(deduplicated_samples))

        balanced_samples = select_balanced_samples(
            deduplicated_samples,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        logging.info("Selected %d balanced samples.", len(balanced_samples))

        splits = stratified_split(balanced_samples, seed=args.seed)
        log_split_summary(splits)

        output_dir = args.output_dir
        logging.info("Writing processed files to %s", output_dir)
        write_jsonl(output_dir / "train.jsonl", splits["train"])
        write_jsonl(output_dir / "val.jsonl", splits["val"])
        write_jsonl(output_dir / "eval.jsonl", splits["eval"])

        logging.info("Data preparation completed successfully.")
    except Exception as exc:
        logging.exception("Data preparation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
