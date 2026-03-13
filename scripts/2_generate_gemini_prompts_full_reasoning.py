"""
Prepare PGPS9K problems as LLM prompts for SFT reasoning generation.

For each problem, produces:
  - A formatted prompt with facts, question, type, and answer
  - The image path
  - Metadata (problem id, original fields)

Output: JSON Lines file where each line is one problem ready for LLM inference.
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path

# Add script dir to path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.convert_questions_and_facts import format_facts_functional, convert_question

SYSTEM_PROMPT = """\
You will be shown a diagram image and given visual facts extracted from it. Produce a rigorous step-by-step solution.

STRICT RULES:
1. Inside <think>, create a <facts> block copying the exact visual facts provided, labeled [F1], [F2], etc. Do NOT paraphrase the facts; preserve their exact functional syntax. Do NOT invent new facts.
2. Inside <think>, create a <theorems> block listing the geometric theorems and formulas needed to solve this problem, labeled [T1], [T2], etc.
3. Inside <think>, create a <reasoning> block. Work through the solution in numbered steps (Step 1, Step 2, ...). You MUST weave the [F] and [T] citations naturally into the English text immediately before executing any math equation. Do not just append them at the end of a sentence.
4. Output the final answer inside <answer> tags. Use LaTeX math notation (e.g., \\frac{17}{2}, 30\\sqrt{30}, 2\\pi, \\sin(60)). Do NOT use plain-text math like sin^2(x) or sqrt(3). Do NOT include units of measurement (e.g., cm, ^\circ, square feet, meters, inches, etc.) in the answer.

EXAMPLE INPUT:
Topic: Circumference and Area of Circle
Question: What is the area of the shaded segment shown at the right?
Facts:
[F1] Line(P, R)
[F2] Line(Q, R)
[F3] Line(Q, P)
[F4] Circle(Q, R, P)
[F5] Perpendicular(Line(Q, R), Line(Q, P), Point(Q))
[F6] Length(P, Q) = 4

EXAMPLE OUTPUT:
<think>
<facts>
[F1] Line(P, R)
[F2] Line(Q, R)
[F3] Line(Q, P)
[F4] Circle(Q, R, P)
[F5] Perpendicular(Line(Q, R), Line(Q, P), Point(Q))
[F6] Length(P, Q) = 4
</facts>
<theorems>
[T1] Circle radius: all points on a circle are equidistant from the center.
[T2] Sector area formula: A = (theta/360) * pi * r^2.
[T3] Right triangle area formula: A = (1/2) * base * height.
[T4] Circular segment area: segment = sector area - triangle area.
</theorems>
<reasoning>
Step 1: From [F4], Q is the center of the circle and R, P lie on it. By [T1] and the given length in [F6], the radius is QR = QP = 4.
Step 2: From the perpendicular lines in [F5], angle RQP = 90 degrees.
Step 3: By applying the sector area formula [T2] using the angle from [F5] and the radius from [F6], the area of sector RQP = (90/360) * pi * 4^2 = 4*pi.
Step 4: By applying the right triangle area formula [T3] using the legs from [F5] and [F6], the area of right triangle RQP = (1/2) * 4 * 4 = 8.
Step 5: By [T4], the shaded segment area is the difference between Step 3 and Step 4: 4\pi - 8.
</reasoning>
</think>
<answer>4\pi - 8</answer>"""

USER_TEMPLATE = """\
You are given a geometry problem with a diagram image and a list of visual facts extracted from the diagram.
Refer to the attached diagram image and the facts below to produce a step-by-step mathematical solution.

Topic: {type}
Question: {question}
Facts:
{facts}"""


def build_user_prompt(problem: dict) -> str:
    """Build the user message text for a single problem."""
    facts_str = format_facts_functional(
        problem["parsing_stru_seqs"],
        problem["parsing_sem_seqs"],
    )
    question = convert_question(problem["text"])

    return USER_TEMPLATE.format(
        type=problem["type"],
        question=question,
        facts=facts_str,
    )


def build_messages(problem: dict, image_path: str) -> list:
    """Build a chat-style message list (system + user with image)."""
    user_text = build_user_prompt(problem)

    system_msg = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }

    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_text},
        ],
    }

    return [system_msg, user_msg]


def main():
    parser = argparse.ArgumentParser(description="Prepare PGPS9K SFT prompts")
    parser.add_argument(
        "--data_dir",
        default="data/PGPS9K/PGPS9K_data",
        help="Path to extracted PGPS9K_data directory (relative to project root)",
    )
    parser.add_argument(
        "--split_name",
        default="PGPS9K",
        choices=["PGPS9K", "Geometry3K"],
        help="Which split to use",
    )
    parser.add_argument(
        "--partition",
        default="train",
        choices=["train", "val", "test"],
        help="Partition to process",
    )
    parser.add_argument(
        "--output",
        default="data/PGPS9K/sft_prompts.jsonl",
        help="Output JSONL file path (relative to project root)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent.parent
    data_dir = str(base / args.data_dir)
    split_path = os.path.join(data_dir, args.split_name, f"{args.partition}.json")
    diagram_dir = os.path.join(data_dir, "Diagram_Visual")

    print(f"Loading {split_path}...")
    with open(split_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} problems")

    count = 0
    output_path = str(base / args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as out:
        for prob_id, problem in data.items():
            image_path = os.path.join(diagram_dir, problem["diagram"])

            if not os.path.exists(image_path):
                print(f"  Warning: missing image {image_path}, skipping {prob_id}")
                continue

            user_prompt = build_user_prompt(problem)
            messages = build_messages(problem, image_path)

            record = {
                "id": prob_id,
                "image": image_path,
                "prompt": user_prompt,
                "messages": messages,
                "answer": problem["answer"],
                "type": problem["type"],
                "diagram": problem["diagram"],
                # Keep original fields for reference
                "parsing_stru_seqs": problem["parsing_stru_seqs"],
                "parsing_sem_seqs": problem["parsing_sem_seqs"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} prompts to {output_path}")

    # Print a sample
    with open(output_path) as f:
        sample = json.loads(f.readline())
    print("\n=== SYSTEM PROMPT ===")
    print(sample["messages"][0]["content"][:200] + "...")
    print("\n=== USER PROMPT ===")
    print(sample["messages"][1]["content"][1]["text"])
    print(f"\nImage: {sample['image']}")


if __name__ == "__main__":
    main()
