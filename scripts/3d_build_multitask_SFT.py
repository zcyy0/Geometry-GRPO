#!/usr/bin/env python3
"""Build multi-task SFT dataset from simplified responses.

Generates 3 training tasks per example:
  Task 1 (visual grounding):  image → numbered NL facts list
  Task 2 (reasoning):         question + golden NL facts → reasoning + answer
  Task 3 (end-to-end):        image + question → reasoning + answer

Usage:
    python scripts/PGPS9K/3d_build_multitask_sft.py
    python scripts/PGPS9K/3d_build_multitask_sft.py --dry_run
"""

import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts" / "PGPS9K"))

from importlib.machinery import SourceFileLoader
_simplify = SourceFileLoader(
    "simplify", str(BASE / "scripts/PGPS9K/3c_simplify_sft_responses.py")
).load_module()

DEFAULT_INPUT = BASE / "data/PGPS9K/sft_responses_train_1500_simplified.jsonl"
DEFAULT_OUTPUT = BASE / "data/PGPS9K/sft_multitask_train.jsonl"

# ---------------------------------------------------------------------------
# System prompts for each task
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TASK1 = """\
You will be shown a geometry diagram. List the visual facts you observe as a numbered list. Include points, lines, angles, lengths, and geometric relationships visible in the diagram."""

SYSTEM_PROMPT_TASK2 = """\
You are given a geometry question and a list of visual facts from the diagram. Solve the problem step by step.

Wrap your reasoning inside <think>...</think> tags. In your reasoning, use the provided facts and apply relevant geometric theorems. Then give your final answer inside <answer>...</answer> tags using LaTeX notation, without units."""

SYSTEM_PROMPT_TASK3 = """\
You will be shown a geometry diagram image and a question. Solve it step by step.

Wrap your reasoning inside <think>...</think> tags. In your reasoning, describe the relevant geometric facts you observe in the diagram and the theorems you apply. Then give your final answer inside <answer>...</answer> tags using LaTeX notation, without units."""


# ---------------------------------------------------------------------------
# Extract and convert facts from original prompt
# ---------------------------------------------------------------------------

def extract_facts_from_prompt(prompt: str) -> list[tuple[str, str]]:
    """Extract (label, raw_text) pairs from the Facts: section of a prompt."""
    facts_m = re.search(r"Facts:\n(.*?)$", prompt, re.DOTALL)
    if not facts_m:
        return []
    facts = []
    for line in facts_m.group(1).strip().split("\n"):
        line = line.strip()
        fm = re.match(r"\[(F\d+)\]\s*(.*)", line)
        if fm:
            facts.append((fm.group(1), fm.group(2).strip()))
    return facts


def extract_question_from_prompt(prompt: str) -> str:
    """Extract the Question line from the prompt."""
    for line in prompt.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("Question:"):
            return stripped
    return prompt.strip()


def facts_to_nl_numbered_list(facts: list[tuple[str, str]]) -> str:
    """Convert functional annotation facts to a numbered NL list.

    E.g.:
        1. Points Q, S, and T are collinear
        2. Line TR
        3. The length of segment QR = 2x+3
    """
    lines = []
    for i, (label, raw) in enumerate(facts, 1):
        nl = _simplify._fact_to_nl(raw)
        # Capitalize first letter
        nl = nl[0].upper() + nl[1:] if nl else nl
        lines.append(f"{i}. {nl}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build examples for each task
# ---------------------------------------------------------------------------

def build_task1(row: dict, facts_nl_list: str) -> dict:
    """Task 1: image → numbered NL facts."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TASK1},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text", "text": "List the geometric facts visible in this diagram."},
            ],
        },
        {
            "role": "assistant",
            "content": facts_nl_list,
        },
    ]
    return {
        "messages": messages,
        "images": [row["image"]],
        "task": "visual_grounding",
        "id": row["id"],
    }


def build_task2(row: dict, question: str, facts_nl_list: str) -> dict:
    """Task 2: question + golden facts → reasoning + answer."""
    user_text = f"{question}\n\nFacts:\n{facts_nl_list}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TASK2},
        {
            "role": "user",
            "content": user_text,
        },
        {
            "role": "assistant",
            "content": row["response"],
        },
    ]
    return {
        "messages": messages,
        "images": [],
        "task": "reasoning",
        "id": row["id"],
    }


def build_task3(row: dict, question: str) -> dict:
    """Task 3: image + question → reasoning + answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TASK3},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": row["response"],
        },
    ]
    return {
        "messages": messages,
        "images": [row["image"]],
        "task": "end_to_end",
        "id": row["id"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build multi-task SFT dataset")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Load data
    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} examples from {args.input.name}")

    # Generate multi-task examples
    all_examples = []
    skipped = 0
    for row in rows:
        # Extract facts and question from original prompt
        facts = extract_facts_from_prompt(row["prompt"])
        question = extract_question_from_prompt(row["prompt"])

        if not facts:
            skipped += 1
            continue

        facts_nl_list = facts_to_nl_numbered_list(facts)

        # Build 3 tasks
        all_examples.append(build_task1(row, facts_nl_list))
        all_examples.append(build_task2(row, question, facts_nl_list))
        all_examples.append(build_task3(row, question))

    # Stats
    task_counts = {}
    for ex in all_examples:
        task_counts[ex["task"]] = task_counts.get(ex["task"], 0) + 1

    print(f"\nGenerated {len(all_examples)} total examples:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    if skipped:
        print(f"  Skipped (no facts): {skipped}")

    # Show examples
    for task_name in ["visual_grounding", "reasoning", "end_to_end"]:
        ex = next(e for e in all_examples if e["task"] == task_name)
        print(f"\n{'='*60}")
        print(f"Task: {task_name} (id: {ex['id']})")
        print(f"{'='*60}")
        print(f"System: {ex['messages'][0]['content'][:100]}...")
        user_msg = ex["messages"][1]["content"]
        if isinstance(user_msg, list):
            text_parts = [p["text"] for p in user_msg if p.get("type") == "text"]
            print(f"User: [image] + {text_parts[0][:200]}")
        else:
            print(f"User: {user_msg[:300]}")
        assistant = ex["messages"][2]["content"]
        print(f"Assistant: {assistant[:300]}...")

    if not args.dry_run:
        with open(args.output, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\nWritten to {args.output}")
    else:
        print("\n(dry run — no file written)")


if __name__ == "__main__":
    main()
