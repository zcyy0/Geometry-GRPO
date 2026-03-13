#!/usr/bin/env python3
"""Evaluate SFT checkpoint on PGPS9K validation set.

Runs inference with a LoRA checkpoint and computes:
1. Answer accuracy
2. Format compliance (<think>, <answer> tags, step numbering)
3. Output token lengths

Supports two modes:
  - Single-task (default): image + question → reasoning + answer
  - Multi-task (--multitask): evaluates all 3 tasks separately
    Task 1 (visual grounding): image → numbered NL facts
    Task 2 (reasoning): question + golden facts → reasoning + answer
    Task 3 (end-to-end): image + question → reasoning + answer

Usage:
    python scripts/PGPS9K/5_eval_sft_checkpoint.py --checkpoint outputs/PGPS9K/sft/final
    python scripts/PGPS9K/5_eval_sft_checkpoint.py --multitask --checkpoint outputs/PGPS9K/sft/final
    python scripts/PGPS9K/5_eval_sft_checkpoint.py  # baseline (no checkpoint)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.compare_answer import extract_answer, parse_numeric, parse_all_numeric, check_answer

# ---------------------------------------------------------------------------
# Paths & prompts
# ---------------------------------------------------------------------------
BASE = project_root
DEFAULT_VAL = BASE / "data/PGPS9K/sft_responses_val_simplified.jsonl"
DEFAULT_MULTITASK_VAL = BASE / "data/PGPS9K/sft_multitask_val.jsonl"

# Must match the system prompt used during SFT training
SYSTEM_PROMPT = """\
You will be shown a geometry diagram image and a question. Solve it step by step.

Wrap your reasoning inside <think>...</think> tags. In your reasoning, describe the relevant geometric facts you observe in the diagram and the theorems you apply. Then give your final answer inside <answer>...</answer> tags using LaTeX notation, without units."""


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
def extract_section(text: str, tag: str) -> str | None:
    """Extract content from a section, supporting both XML tags and header-based format.

    XML format:    <facts>...</facts>
    Header format: Visual Facts:\n...\n\nTheorems:\n...
    """
    # Try XML tags first
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fall back to header-based extraction inside <think>...</think>
    think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not think_m:
        return None
    think_content = think_m.group(1)

    header_map = {
        "facts": "Visual Facts:",
        "theorems": "Theorems:",
        "reasoning": "Reasoning:",
    }
    header = header_map.get(tag)
    if not header:
        return None

    # All possible section headers in order
    all_headers = ["Visual Facts:", "Theorems:", "Reasoning:"]
    start_idx = think_content.find(header)
    if start_idx < 0:
        return None
    start_idx += len(header)

    # Find the start of the next section (or end of think content)
    end_idx = len(think_content)
    for h in all_headers:
        if h == header:
            continue
        h_idx = think_content.find(h, start_idx)
        if h_idx >= 0 and h_idx < end_idx:
            end_idx = h_idx

    return think_content[start_idx:end_idx].strip()


def extract_labels(text: str, prefix: str) -> list[str]:
    """Extract all [F1], [T2], etc. labels from text."""
    return re.findall(rf"\[{prefix}\d+\]", text)


def extract_label_definitions(text: str, prefix: str) -> dict[str, str]:
    """Extract label -> content, e.g. {'[F1]': 'Line(A, B)'}."""
    defs = {}
    for m in re.finditer(
        rf"(\[{prefix}\d+\])\s*(.+?)(?=\[{prefix}\d+\]|$)", text, re.DOTALL
    ):
        defs[m.group(1)] = m.group(2).strip()
    return defs


def extract_facts_from_prompt(prompt: str) -> dict[str, str]:
    """Extract ground truth facts from the raw prompt text (before stripping)."""
    facts_m = re.search(r"Facts:\n(.*?)$", prompt, re.DOTALL)
    if not facts_m:
        return {}
    return extract_label_definitions(facts_m.group(1), "F")


def strip_facts_from_prompt(prompt: str) -> str:
    """Strip the preamble, Topic, and Facts section from the prompt, keeping Question only."""
    for line in prompt.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("Question:"):
            return stripped
    return prompt.strip()


def normalize_fact(fact_text: str) -> str:
    """Normalize a fact for comparison.

    Handles commutativity:
    - Line(A, B) = Line(B, A)
    - Length(A, B) = Length(B, A)
    - Angle(A, B, C) = Angle(C, B, A)  (vertex stays, sort first/last)
    - ArcDegree(A, B) = ArcDegree(B, A)
    - ArcLength(A, B) = ArcLength(B, A)
    - Parallel/Perpendicular: normalize inner Line() args, sort lines
    - Collinear: sort all args
    Does NOT normalize order-sensitive types: Circle.
    """
    s = re.sub(r"\s+", " ", fact_text).strip()

    # Collinear(A, B, C) → sort all args
    def sort_all_args(match):
        func = match.group(1)
        args = [a.strip() for a in match.group(2).split(",")]
        args.sort()
        return f"{func}({', '.join(args)})"

    s = re.sub(r"(Collinear)\(([^)]+)\)", sort_all_args, s)

    # Line(A, B), Length(A, B) → sort args
    def sort_binary_args(match):
        func = match.group(1)
        args = [a.strip() for a in match.group(2).split(",")]
        if len(args) == 2:
            args.sort()
        return f"{func}({', '.join(args)})"

    s = re.sub(r"(Line|Length|ArcDegree|ArcLength|MajorArcDegree|MajorArcLength)\(([^)]+)\)", sort_binary_args, s)

    # Angle(A, B, C) → keep vertex (middle), sort first and last
    def normalize_angle(match):
        args = [a.strip() for a in match.group(1).split(",")]
        if len(args) == 3:
            first, vertex, last = args
            ends = sorted([first, last])
            return f"Angle({ends[0]}, {vertex}, {ends[1]})"
        return match.group(0)

    s = re.sub(r"Angle\(([^)]+)\)", normalize_angle, s)

    # Parallel(Line(...), Line(...)) → sort the normalized Line() terms
    def normalize_parallel(match):
        func = match.group(1)  # Parallel or Perpendicular
        inner = match.group(2)
        # Extract all Line(...) terms and any Point(...) term
        lines = re.findall(r"Line\([^)]+\)", inner)
        point = re.search(r"Point\([^)]+\)", inner)
        lines.sort()
        parts = ", ".join(lines)
        if point:
            parts += f", {point.group(0)}"
        return f"{func}({parts})"

    s = re.sub(r"(Parallel|Perpendicular)\((.+)\)", normalize_parallel, s)

    # Normalize equality: sort sides so "Length(C,D) = Length(A,B)" matches
    # "Length(A,B) = Length(C,D)"
    if " = " in s:
        parts = [p.strip() for p in s.split(" = ")]
        s = " = ".join(sorted(parts))

    return s


def normalize_theorem(theorem_text: str) -> str:
    """Normalize theorem text for comparison."""
    s = theorem_text.lower().strip()
    s = re.sub(r"[.,:;!]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# Per-example metrics
# ---------------------------------------------------------------------------
def compute_metrics_for_one(
    response: str,
    raw_prompt: str,
    gold_answer: str,
    ref_response: str | None = None,
) -> dict:
    """Compute metrics for a single example (simplified format)."""
    m = {}

    # ── 1. Accuracy ──
    gold_val = parse_numeric(str(gold_answer))
    pred_text = extract_answer(response)
    pred_vals = parse_all_numeric(pred_text) if pred_text else []
    m["correct"] = bool(
        gold_val is not None and pred_vals and check_answer(pred_vals, gold_val)
    )
    m["answer_extracted"] = pred_text is not None

    # ── 2. Format compliance (simplified) ──
    has_think = "<think>" in response
    has_answer = "<answer>" in response
    m["has_think"] = has_think
    m["has_answer"] = has_answer

    # Extract think content for step numbering check
    think_m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    think_text = think_m.group(1) if think_m else ""
    m["has_step_numbering"] = bool(re.search(r"Step\s+\d+", think_text))
    m["format_compliant"] = has_think and has_answer and m["has_step_numbering"]

    return m


# ---------------------------------------------------------------------------
# Multi-task metrics
# ---------------------------------------------------------------------------
def compute_task1_metrics(response: str, gold_response: str) -> dict:
    """Metrics for visual grounding: numbered NL facts list."""
    m = {}

    # Count predicted facts (lines matching "N. ...")
    pred_lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    pred_facts = [l for l in pred_lines if re.match(r"\d+\.\s", l)]
    m["num_facts_pred"] = len(pred_facts)

    # Count gold facts
    gold_lines = [l.strip() for l in gold_response.strip().split("\n") if l.strip()]
    gold_facts = [l for l in gold_lines if re.match(r"\d+\.\s", l)]
    m["num_facts_gold"] = len(gold_facts)

    # Format: is it a numbered list?
    m["is_numbered_list"] = len(pred_facts) > 0 and len(pred_facts) == len(pred_lines)

    # No extra tags that shouldn't be there
    m["no_think_tag"] = "<think>" not in response
    m["no_answer_tag"] = "<answer>" not in response
    m["format_compliant"] = m["is_numbered_list"] and m["no_think_tag"] and m["no_answer_tag"]

    return m


def compute_task23_metrics(response: str, gold_answer_text: str) -> dict:
    """Metrics for reasoning / end-to-end: accuracy + format compliance."""
    m = {}

    # Accuracy
    gold_val = parse_numeric(str(gold_answer_text))
    pred_text = extract_answer(response)
    pred_vals = parse_all_numeric(pred_text) if pred_text else []
    m["correct"] = bool(
        gold_val is not None and pred_vals and check_answer(pred_vals, gold_val)
    )
    m["answer_extracted"] = pred_text is not None

    # Format compliance
    has_think = "<think>" in response
    has_answer = "<answer>" in response
    m["has_think"] = has_think
    m["has_answer"] = has_answer
    think_m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    think_text = think_m.group(1) if think_m else ""
    m["has_step_numbering"] = bool(re.search(r"Step\s+\d+", think_text))
    m["format_compliant"] = has_think and has_answer and m["has_step_numbering"]

    return m


# ---------------------------------------------------------------------------
# Multi-task data loading
# ---------------------------------------------------------------------------
def load_multitask_val_data(path: Path) -> list[dict]:
    """Load multi-task validation data (pre-built messages)."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gold_response = obj["messages"][2]["content"]

            # For tasks 2 & 3, extract gold answer from the assistant response
            gold_answer = extract_answer(gold_response)

            examples.append({
                "id": obj["id"],
                "task": obj["task"],
                "images": obj["images"],
                "system_prompt": obj["messages"][0]["content"],
                "user_content": obj["messages"][1]["content"],
                "gold_response": gold_response,
                "gold_answer": gold_answer,
            })
    return examples


def build_multitask_conversation(ex: dict) -> list[dict]:
    """Convert a multi-task example to a vLLM conversation (system + user only)."""
    user_content = ex["user_content"]

    # Convert image format: training uses {"type": "image", "image": path}
    # but vLLM expects {"type": "image_url", "image_url": {"url": "file://..."}}
    if isinstance(user_content, list):
        converted = []
        for part in user_content:
            if part.get("type") == "image":
                converted.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{part['image']}"},
                })
            else:
                converted.append(part)
        user_content = converted

    return [
        {"role": "system", "content": ex["system_prompt"]},
        {"role": "user", "content": user_content},
    ]


def run_multitask_inference(
    llm: LLM,
    examples: list[dict],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
) -> list[tuple[str, int]]:
    """Run batched vLLM inference on multi-task examples."""
    conversations = [build_multitask_conversation(ex) for ex in examples]

    # Sanity check — one per task
    shown = set()
    for ex, conv in zip(examples, conversations):
        if ex["task"] not in shown:
            shown.add(ex["task"])
            user_msg = conv[1]["content"]
            if isinstance(user_msg, list):
                text_parts = [p.get("text", "[image]") for p in user_msg]
                user_preview = " | ".join(text_parts)[:150]
            else:
                user_preview = str(user_msg)[:150]
            print(f"  [{ex['task']}] {ex['id']}: {user_preview}...")

    print(f"\nRunning vLLM inference on {len(conversations)} examples...")
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outputs]


def print_multitask_report(task_results: dict[str, dict]):
    """Print per-task evaluation report."""
    print(f"\n{'='*70}")
    print(f"Multi-Task SFT Evaluation Report")
    print(f"{'='*70}")

    for task_name in ["visual_grounding", "reasoning", "end_to_end"]:
        tr = task_results.get(task_name)
        if not tr:
            continue
        metrics = tr["metrics"]
        n = tr["count"]

        print(f"\n--- Task: {task_name} ({n} examples) ---")

        if task_name == "visual_grounding":
            print(f"  Format compliant (numbered list): {metrics['format_compliant']:.1%}")
            print(f"  Avg facts predicted:              {metrics['avg_facts_pred']:.1f}")
            print(f"  Avg facts gold:                   {metrics['avg_facts_gold']:.1f}")
        else:
            print(f"  Accuracy:                         {metrics['correct']:.1%}")
            print(f"  Answer extracted:                 {metrics['answer_extracted']:.1%}")
            print(f"  Format compliant:                 {metrics['format_compliant']:.1%}")
            print(f"  Has <think>:                      {metrics['has_think']:.1%}")
            print(f"  Has <answer>:                     {metrics['has_answer']:.1%}")
            print(f"  Has step numbering:               {metrics['has_step_numbering']:.1%}")

        tok = metrics["output_tokens"]
        print(f"  Tokens — min: {tok['min']}, median: {tok['median']}, "
              f"max: {tok['max']}, mean: {tok['mean']:.0f}, hit_limit: {tok['hit_limit']}")


# ---------------------------------------------------------------------------
# Data loading (single-task)
# ---------------------------------------------------------------------------
def load_val_data(path: Path) -> list[dict]:
    """Load validation data with prompts and reference responses."""
    examples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            raw_prompt = obj["prompt"]
            examples.append({
                "id": obj["id"],
                "image": obj["image"],
                "type": obj.get("type", ""),
                "answer": obj["answer"],
                "raw_prompt": raw_prompt,  # kept for ground truth fact extraction
                "prompt": strip_facts_from_prompt(raw_prompt),  # for inference
                "ref_response": obj["response"],
            })
    return examples


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    llm: LLM,
    examples: list[dict],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
) -> list[str]:
    """Run batched vLLM inference and return response strings."""
    conversations = []
    for ex in examples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{ex['image']}"},
                    },
                    {"type": "text", "text": ex["prompt"]},
                ],
            },
        ]
        conversations.append(messages)

    # Sanity check
    print(f"\n--- Sanity check (first example) ---")
    print(f"  ID: {examples[0]['id']}")
    print(f"  Image: {examples[0]['image']}")
    print(f"  Prompt: {examples[0]['prompt'][:150]}...")
    print(f"  Gold: {examples[0]['answer']}")
    print()

    print(f"Running vLLM inference on {len(conversations)} examples...")
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outputs]


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------
def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Aggregate per-example metrics into summary."""
    n = len(all_metrics)
    s = {}

    # Bool -> rate
    for key in [
        "correct", "answer_extracted", "has_think", "has_answer",
        "has_step_numbering", "format_compliant",
    ]:
        s[key] = sum(m[key] for m in all_metrics) / n

    return s


def print_report(summary: dict, by_type: dict):
    """Print formatted evaluation report."""
    s = summary

    print(f"\n{'='*70}")
    print(f"SFT Checkpoint Evaluation Report")
    print(f"{'='*70}")

    print(f"\n--- 1. Answer Accuracy ---")
    print(f"  Accuracy:              {s['correct']:.1%}")
    print(f"  Answer extracted:      {s['answer_extracted']:.1%}")

    print(f"\n--- 2. Format Compliance ---")
    print(f"  Full compliance:       {s['format_compliant']:.1%}")
    print(f"  Has <think>:           {s['has_think']:.1%}")
    print(f"  Has <answer>:          {s['has_answer']:.1%}")
    print(f"  Has step numbering:    {s['has_step_numbering']:.1%}")

    print(f"\n--- 3. Output Token Length ---")
    tok = s["output_tokens"]
    print(f"  Min tokens:            {tok['min']}")
    print(f"  Max tokens:            {tok['max']}")
    print(f"  Mean tokens:           {tok['mean']:.0f}")
    print(f"  Median tokens:         {tok['median']}")
    print(f"  Hit token limit:       {tok['hit_limit']}")

    print(f"\n--- Accuracy by Type ---")
    print(f"  {'Type':<45} {'Acc':>8} {'N':>6}")
    print(f"  {'-'*45} {'-'*8} {'-'*6}")
    for t in sorted(
        by_type.keys(),
        key=lambda t: by_type[t]["correct"] / max(by_type[t]["total"], 1),
    ):
        bt = by_type[t]
        acc = bt["correct"] / bt["total"] if bt["total"] else 0
        print(f"  {t:<45} {acc:>7.1%} {bt['total']:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SFT checkpoint on PGPS9K"
    )
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to LoRA checkpoint (omit for baseline)",
    )
    parser.add_argument("--val_data", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--multitask", action="store_true",
                        help="Evaluate on multi-task val set (3 tasks)")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--max_pixels", type=int, default=1003520)
    args = parser.parse_args()

    # Load data
    if args.multitask:
        mt_val_path = args.val_data if "multitask" in str(args.val_data) else DEFAULT_MULTITASK_VAL
        examples = load_multitask_val_data(mt_val_path)
        if args.max_examples:
            examples = examples[: args.max_examples]
        print(f"Loaded {len(examples)} multi-task validation examples")
        task_counts = {}
        for ex in examples:
            task_counts[ex["task"]] = task_counts.get(ex["task"], 0) + 1
        for t, c in sorted(task_counts.items()):
            print(f"  {t}: {c}")
    else:
        examples = load_val_data(args.val_data)
        if args.max_examples:
            examples = examples[: args.max_examples]
        print(f"Loaded {len(examples)} validation examples")

    # Allowed media path for vLLM
    if args.multitask:
        all_images = [img for ex in examples for img in ex["images"]]
    else:
        all_images = [ex["image"] for ex in examples]
    image_dirs = {str(Path(img).parent) for img in all_images if img}
    data_dir = os.path.commonpath(list(image_dirs)) if image_dirs else "/"

    # Load model
    ckpt_label = Path(args.checkpoint).name if args.checkpoint else "baseline"
    print(f"Loading model: {args.model_id} (checkpoint: {ckpt_label})")

    llm_kwargs = dict(
        model=args.model_id,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=data_dir,
        mm_processor_kwargs={
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
    )
    lora_request = None
    if args.checkpoint:
        ckpt_path = str(BASE / args.checkpoint)
        print(f"  with LoRA: {ckpt_path}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32
        lora_request = LoRARequest("eval_lora", 1, ckpt_path)

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=0, seed=42,
    )

    if args.multitask:
        # ── Multi-task evaluation ──
        print("\nSanity check (one per task):")
        responses = run_multitask_inference(llm, examples, sampling_params, lora_request)

        # Compute per-task metrics
        task_examples = defaultdict(list)
        for ex, (response, num_tokens) in zip(examples, responses):
            task_examples[ex["task"]].append((ex, response, num_tokens))

        task_results = {}
        all_per_example = []

        for task_name, items in task_examples.items():
            task_metrics_list = []
            token_lengths = []

            for ex, response, num_tokens in items:
                if task_name == "visual_grounding":
                    met = compute_task1_metrics(response, ex["gold_response"])
                else:
                    met = compute_task23_metrics(response, ex["gold_answer"] or "")
                met["id"] = ex["id"]
                met["task"] = task_name
                met["response"] = response
                met["output_tokens"] = num_tokens
                task_metrics_list.append(met)
                token_lengths.append(num_tokens)

            # Aggregate
            n = len(task_metrics_list)
            tok_summary = {
                "min": min(token_lengths),
                "max": max(token_lengths),
                "mean": sum(token_lengths) / n,
                "median": sorted(token_lengths)[n // 2],
                "hit_limit": sum(1 for t in token_lengths if t >= args.max_new_tokens),
            }

            if task_name == "visual_grounding":
                agg = {
                    "format_compliant": sum(m["format_compliant"] for m in task_metrics_list) / n,
                    "avg_facts_pred": sum(m["num_facts_pred"] for m in task_metrics_list) / n,
                    "avg_facts_gold": sum(m["num_facts_gold"] for m in task_metrics_list) / n,
                    "output_tokens": tok_summary,
                }
            else:
                agg = {
                    "correct": sum(m["correct"] for m in task_metrics_list) / n,
                    "answer_extracted": sum(m["answer_extracted"] for m in task_metrics_list) / n,
                    "format_compliant": sum(m["format_compliant"] for m in task_metrics_list) / n,
                    "has_think": sum(m["has_think"] for m in task_metrics_list) / n,
                    "has_answer": sum(m["has_answer"] for m in task_metrics_list) / n,
                    "has_step_numbering": sum(m["has_step_numbering"] for m in task_metrics_list) / n,
                    "output_tokens": tok_summary,
                }

            task_results[task_name] = {"count": n, "metrics": agg}
            all_per_example.extend(task_metrics_list)

        print_multitask_report(task_results)

        # Save
        output_path = args.output or f"outputs/PGPS9K/eval_sft_multitask_{ckpt_label}.json"
        output_full = str(BASE / output_path)
        os.makedirs(os.path.dirname(output_full), exist_ok=True)

        with open(output_full, "w") as f:
            json.dump(
                {"task_results": task_results, "per_example": all_per_example},
                f, indent=2, ensure_ascii=False, default=str,
            )
        print(f"\nResults saved to {output_path}")

    else:
        # ── Single-task evaluation ──
        responses = run_inference(llm, examples, sampling_params, lora_request)

        all_metrics = []
        by_type = defaultdict(lambda: {"correct": 0, "total": 0})
        token_lengths = []

        for ex, (response, num_tokens) in zip(examples, responses):
            met = compute_metrics_for_one(
                response=response,
                raw_prompt=ex["raw_prompt"],
                gold_answer=ex["answer"],
                ref_response=ex["ref_response"],
            )
            met["id"] = ex["id"]
            met["type"] = ex["type"]
            met["response"] = response
            met["output_tokens"] = num_tokens
            all_metrics.append(met)
            token_lengths.append(num_tokens)

            by_type[ex["type"]]["total"] += 1
            if met["correct"]:
                by_type[ex["type"]]["correct"] += 1

        token_summary = {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
            "median": sorted(token_lengths)[len(token_lengths) // 2],
            "hit_limit": sum(1 for t in token_lengths if t >= args.max_new_tokens),
        }

        summary = aggregate_metrics(all_metrics)
        summary["output_tokens"] = token_summary
        print_report(summary, dict(by_type))

        # Save
        output_path = args.output or f"outputs/PGPS9K/eval_sft_{ckpt_label}.json"
        output_full = str(BASE / output_path)
        os.makedirs(os.path.dirname(output_full), exist_ok=True)

        with open(output_full, "w") as f:
            json.dump(
                {"summary": summary, "by_type": dict(by_type), "per_example": all_metrics},
                f, indent=2, ensure_ascii=False, default=str,
            )
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
