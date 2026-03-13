"""Evaluate Qwen2.5-VL-3B on the PGPS9K geometry dataset using vLLM.

Runs greedy decoding on the PGPS9K test set (or train/val) and reports
accuracy overall and broken down by geometry type.

Usage:
    # Evaluate baseline model
    python scripts/PGPS9K/1_eval_pgps9k_baseline.py

    # Evaluate with a LoRA checkpoint
    python scripts/PGPS9K/1_eval_pgps9k_baseline.py --checkpoint outputs/grpo_curriculum/phase3_final

    # Evaluate on val split
    python scripts/PGPS9K/1_eval_pgps9k_baseline.py --partition val
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from vllm import LLM, SamplingParams

# Add project root and script dir to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.convert_questions_and_facts import convert_question
from utils.compare_answer import extract_answer, parse_numeric, parse_all_numeric, check_answer, TOLERANCE


def score_one(gold: str, pred_response: str) -> tuple[bool, str | None, float | None, float | None]:
    """Score a single prediction against gold via numeric evaluation.

    Returns (is_correct, pred_answer_text, pred_value, gold_value).
    """
    gold_value = parse_numeric(str(gold))
    if gold_value is None:
        return False, None, None, None

    pred_text = extract_answer(pred_response)
    if pred_text is None:
        return False, None, None, gold_value

    pred_values = parse_all_numeric(pred_text)
    if not pred_values:
        return False, pred_text, None, gold_value

    return check_answer(pred_values, gold_value), pred_text, pred_values[0] if len(pred_values) == 1 else pred_values, gold_value

# ── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You will be shown a diagram and a geometry question. "
    "Think step by step, then give the final answer.\n"
    "Return ONLY in this exact format: <think>...</think> <answer>...</answer>.\n"
    "Output the final answer inside <answer> tags. Use LaTeX math notation "
    "(e.g., \\frac{17}{2}, 30\\sqrt{30}, 2\\pi, \\sin(60)). "
    "Do NOT use plain-text math like sin^2(x) or sqrt(3). "
    "Do NOT include units (no degrees, feet, cm, etc.)."
)


def build_eval_prompt(problem: dict) -> str:
    """Build the user-facing question text for evaluation (image + question only)."""
    return convert_question(problem["text"])


# ── Data loading ────────────────────────────────────────────────────────────

def load_pgps9k(data_dir: str, partition: str,
                max_examples: int = 0) -> list[dict]:
    """Load PGPS9K problems with resolved image paths."""
    split_path = os.path.join(data_dir, "PGPS9K", f"{partition}.json")
    diagram_dir = os.path.join(data_dir, "Diagram_Visual")

    with open(split_path) as f:
        data = json.load(f)

    examples = []
    for prob_id, problem in data.items():
        image_path = os.path.join(diagram_dir, problem["diagram"])
        if not os.path.exists(image_path):
            continue
        examples.append({
            "id": prob_id,
            "image_path": image_path,
            "problem": problem,
            "gt": problem["answer"],
            "type": problem["type"],
        })
        if max_examples and len(examples) >= max_examples:
            break

    return examples


# ── Inference ───────────────────────────────────────────────────────────────

def run_inference(llm: LLM, examples: list[dict],
                  sampling_params: SamplingParams) -> list[dict]:
    """Run batched vLLM inference on all examples and return results."""

    # Build all chat messages
    conversations = []
    for ex in examples:
        user_text = build_eval_prompt(ex["problem"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{ex['image_path']}"}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        conversations.append(messages)

    # Log first example for sanity check
    print(f"\n--- First example sanity check ---")
    print(f"  ID: {examples[0]['id']}")
    print(f"  Image: {examples[0]['image_path']}")
    print(f"  Question: {conversations[0][1]['content'][1]['text']}")
    print(f"  GT answer: {examples[0]['gt']}")
    print()

    # Batch generate
    print(f"Running vLLM batch inference on {len(conversations)} examples...")
    outputs = llm.chat(conversations, sampling_params=sampling_params)

    # Score results
    results = []
    correct_so_far = 0
    for i, (ex, output) in enumerate(zip(examples, outputs)):
        response = output.outputs[0].text
        user_text = build_eval_prompt(ex["problem"])

        is_correct, pred_text, pred_value, gold_value = score_one(ex["gt"], response)
        correct_so_far += is_correct

        # Log first 3 examples in detail
        if i < 3:
            status = "CORRECT" if is_correct else "WRONG"
            print(f"\n--- Example {i+1} [{status}] ---")
            print(f"  ID: {ex['id']} | Type: {ex['type']}")
            print(f"  Question: {user_text}")
            print(f"  GT: {ex['gt']} ({gold_value}) | Pred: {pred_text} ({pred_value})")
            resp_preview = response[:300] + "..." if len(response) > 300 else response
            print(f"  Response: {resp_preview}")

        # Running accuracy every 50 examples
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(examples)}] running accuracy: {correct_so_far}/{i+1} ({correct_so_far/(i+1):.1%})")

        results.append({
            "id": ex["id"],
            "type": ex["type"],
            "image_path": ex["image_path"],
            "gt": ex["gt"],
            "gt_value": gold_value,
            "pred_answer": pred_text,
            "pred_value": pred_value,
            "is_correct": is_correct,
            "response": response,
            "prompt": user_text,
        })

    return results


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute overall and per-type accuracy."""
    metrics = {}

    # Overall
    total = len(results)
    correct = sum(r["is_correct"] for r in results)
    metrics["overall_accuracy"] = correct / total if total else 0
    metrics["total"] = total
    metrics["correct"] = correct

    # Parse success rate (answer extracted AND evaluable to a number)
    parsed = sum(1 for r in results if r["pred_value"] is not None)
    metrics["parse_success_rate"] = parsed / total if total else 0

    # Per type
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    type_metrics = {}
    for t, recs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        c = sum(r["is_correct"] for r in recs)
        type_metrics[t] = {
            "accuracy": c / n if n else 0,
            "correct": c,
            "total": n,
        }
    metrics["by_type"] = type_metrics

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 70}")
    print(f"PGPS9K Evaluation Results")
    print(f"{'=' * 70}")
    print(f"Overall Accuracy: {metrics['correct']}/{metrics['total']} "
          f"({metrics['overall_accuracy']:.1%})")
    print(f"Parse Success Rate: {metrics['parse_success_rate']:.1%}")
    print(f"\n{'Type':<45} {'Acc':>8} {'Correct':>8} {'Total':>6}")
    print(f"{'-' * 45} {'-' * 8} {'-' * 8} {'-' * 6}")
    for t, m in metrics["by_type"].items():
        print(f"{t:<45} {m['accuracy']:>7.1%} {m['correct']:>8} {m['total']:>6}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate on PGPS9K")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint (omit for baseline)")
    parser.add_argument("--data_dir", type=str,
                        default="data/PGPS9K/PGPS9K_data",
                        help="Path to PGPS9K_data directory")
    parser.add_argument("--partition", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Limit number of examples (0 = all)")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max total sequence length (input + output tokens)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSONL file")
    parser.add_argument("--min_pixels", type=int, default=3136,
                        help="Min image pixels (default 3136 = 56x56)")
    parser.add_argument("--max_pixels", type=int, default=1003520,
                        help="Max image pixels (default 1003520 = 28*28*1280)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent.parent
    data_dir = str(base / args.data_dir)

    ckpt_label = Path(args.checkpoint).name if args.checkpoint else "baseline"

    # Load data
    print(f"Loading PGPS9K {args.partition}...")
    examples = load_pgps9k(
        data_dir, args.partition, args.max_examples,
    )
    print(f"Loaded {len(examples)} examples")

    # Load model with vLLM
    print(f"Loading model with vLLM: {args.model_id}")
    print(f"  Image resolution: min_pixels={args.min_pixels}, max_pixels={args.max_pixels}")
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
    if args.checkpoint:
        ckpt_path = str(base / args.checkpoint)
        print(f"  with LoRA: {ckpt_path}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["lora_modules"] = [{"name": "eval_lora", "path": ckpt_path}]

    llm = LLM(**llm_kwargs)

    # Sampling params (greedy)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0,
        seed=42,
    )

    # Run inference
    print("Running evaluation (image + question only)...")
    results = run_inference(llm, examples, sampling_params)

    # Compute and print metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = f"outputs/PGPS9K/eval_results_{ckpt_label}_{args.partition}.jsonl"

    output_full = str(base / output_path)
    os.makedirs(os.path.dirname(output_full), exist_ok=True)
    with open(output_full, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {output_path}")

    # Save summary
    summary_path = output_full.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Summary saved to {summary_path.replace(str(base) + '/', '')}")

    # Save wrong predictions for analysis
    wrong = [r for r in results if not r["is_correct"]]
    wrong_path = output_full.replace(".jsonl", "_wrong.json")
    with open(wrong_path, "w") as f:
        json.dump(wrong, f, indent=2, ensure_ascii=False)
    print(f"Wrong predictions saved to {wrong_path.replace(str(base) + '/', '')} ({len(wrong)} examples)")



if __name__ == "__main__":
    main()
