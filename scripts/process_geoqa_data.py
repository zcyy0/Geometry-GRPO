#!/usr/bin/env python3
"""
Convert geoqa MCQ questions to open-ended questions AND normalize the
extracted answers in a single pass.

Input:  geoqa_only.jsonl   (raw MCQ data)
Output: geoqa_open_normalized.jsonl  (open questions with $-wrapped LaTeX answers)
        geoqa_open_normalized_preview.json  (side-by-side for inspection)

Usage:
    python scripts/geoqa/convert_and_normalize.py
    python scripts/geoqa/convert_and_normalize.py --input /path/to/input.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.extract_answer import normalize_geoqa_ground_truth


# ── Choice block extraction ──────────────────────────────────────────────────

CHOICES_BLOCK_RE = re.compile(
    r"\n?Choices:\s*\n((?:[A-E]:\s*.+(?:\n|$))+)",
    re.IGNORECASE,
)

INDIVIDUAL_CHOICE_RE = re.compile(r"([A-E]):\s*(.+?)(?:\n|$)")


def parse_choices(question: str):
    """Extract choice dict {A: '4', B: '6', ...} and the choices block match."""
    m = CHOICES_BLOCK_RE.search(question)
    if not m:
        return None, None
    choices = {}
    for letter, value in INDIVIDUAL_CHOICE_RE.findall(m.group(0)):
        choices[letter.upper()] = value.strip()
    return choices, m


# ── Instruction removal ─────────────────────────────────────────────────────

PREFIX_PATTERNS = [
    r"Answer\s+with\s+the\s+option'?s?\s+letter\s+from\s+the\s+given\s+choices\s+directly\.?\s*\n?",
    r"Provide\s+your\s+answer\s+solely\s+as\s+the\s+option\s+letter,?\s*without\s+any\s+additional\s+text\.?\s*\n?",
    r"Please\s+choose\s+the\s+correct\s+option\.?\s*\n?",
    r"Choose\s+the\s+correct\s+answer\s+from\s+the\s+given\s+options\.?\s*\n?",
    r"Select\s+the\s+correct\s+option\.?\s*\n?",
    r"Please\s+answer\s+with\s+the\s+letter\s+of\s+the\s+correct\s+option\.?\s*\n?",
    r"Please\s+select\s+the\s+correct\s+answer\s+from\s+the\s+options\s+below\.?\s*\n?",
    r"Please\s+provide\s+the\s+letter\s+of\s+the\s+correct\s+answer\.?\s*\n?",
    r"Simply\s+indicate\s+the\s+correct\s+answer\s+by\s+typing\s+the\s+letter\s+from\s+the\s+given\s+options\.?\s*\n?",
    r"Just\s+indicate\s+the\s+correct\s+answer\s+by\s+typing\s+the\s+letter\.?\s*\n?",
    r"Indicate\s+the\s+correct\s+answer\s+by\s+typing\s+the\s+letter\s+from\s+the\s+given\s+options\.?\s*\n?",
]
PREFIX_RES = [re.compile(p, re.IGNORECASE) for p in PREFIX_PATTERNS]

MCQ_KEYWORDS = {"answer", "letter", "option", "options", "choices"}


def _is_instruction_sentence(sentence: str) -> bool:
    words = set(re.sub(r"[^\w\s]", "", sentence.lower()).split())
    return len(MCQ_KEYWORDS & words) >= 2


def remove_instructions(question: str) -> str:
    for pat in PREFIX_RES:
        question = pat.sub("", question, count=1).lstrip()
    lines = question.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped and _is_instruction_sentence(stripped):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


# ── Filtering ────────────────────────────────────────────────────────────────

# Matches synthetic variable-assignment sentences:
#   "Represent the area as 'x' in relation to ..."
#   "represented as variable 'v' in relation to 'u'"
#   "expressed as variable y in relation to x"
#   "denoted as variable E1, in terms of L, D1, and d"
#   "Express the angle as 'y' in terms of 'x'"
#   "let's denote ... as variable x"
_VARIABLE_PATTERN_RE = re.compile(
    r"(?:represent(?:ed)?|express(?:ed)?|denot(?:ed?|ing))\s+"
    r".*?\b(?:as\s+(?:variable\s+)?['\"]?\w['\"]?|variable\s+['\"]?\w['\"]?)"
    r".*?(?:in\s+(?:relation|terms)\s+(?:to|of)\b)",
    re.IGNORECASE,
)


def should_skip(ex: dict) -> str | None:
    """
    Return a skip reason string if *ex* should be filtered out, else None.
    """
    question = ex.get("question", "")

    # Any choice option contains "或" (Chinese "or" → ambiguous answer)
    choices, _ = parse_choices(question)
    if choices:
        for value in choices.values():
            if "或" in value:
                return "choice_contains_或"

    return None


# ── Variable-phrase cleaning ──────────────────────────────────────────────────

# Trailing "in terms of [X]" / "in relation to [X]" at end of sentence.
# Also handles "denoted in terms of" (without as/by).
# Captures the text after of/to so we can check if it's π (keep) or not (strip).
_IN_TERMS_OF_RE = re.compile(
    r"""
    ,?\s*
    (?:(?:expressed|represented|denoted)\s+(?:(?:as|by)\s+(?:(?:variable\s+)?['\"]?\w+['\"]?\s+)?)?)?
    in\s+(?:terms|relation)\s+(?:of|to)\s+
    (.+?)                     # what follows "of/to"
    \s*([?.]?)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Trailing ", denoted as/by 'X'" / ", represented as/by 'X'" / ", expressed as 'X'"
_DENOTED_TRAILING_RE = re.compile(
    r""",?\s*(?:denoted|represented|expressed)\s+(?:as|by)\s+.+?\s*([?.]?)\s*$""",
    re.IGNORECASE,
)

# Standalone "Denote the ... as 'X'" sentences
_DENOTE_SENTENCE_RE = re.compile(
    r"^(?:Denote|Let'?s?\s+denote)\b.*(?:as|by)\s+['\"]?\w",
    re.IGNORECASE,
)

# "Represent the ... as 'X'" standalone directive (without "in terms of")
_REPRESENT_SENTENCE_RE = re.compile(
    r"^Represent\s+the\s+\w.*?\s+as\s+['\"]?\w['\"]?\s*\.?\s*$",
    re.IGNORECASE,
)


def _strip_in_terms_of(sentence: str) -> str:
    """Strip trailing 'in terms of [X]' unless X contains π."""
    m = _IN_TERMS_OF_RE.search(sentence)
    if not m:
        return sentence
    after_of = m.group(1).strip()
    # Keep if it's solely about π (e.g. "in terms of π"), not mixed with variables
    if "π" in after_of and not re.search(r"['\"]", after_of):
        return sentence
    punct = m.group(2)
    return sentence[: m.start()].rstrip(", ") + punct


def _strip_denoted_trailing(sentence: str) -> str:
    """Strip trailing ', denoted as/by ...' phrases."""
    m = _DENOTED_TRAILING_RE.search(sentence)
    if not m:
        return sentence
    punct = m.group(1)
    return sentence[: m.start()].rstrip(", ") + punct


def _is_meaningless_after_strip(sentence: str) -> bool:
    """Check if a sentence became meaningless after stripping."""
    s = sentence.rstrip(".? ").strip()
    if not s:
        return True
    # "Express the angle as 'y'.", "Represent the area as 'x'."
    if re.match(
        r"^(?:Express|Represent)\s+.*\bas\s+['\"]?\w+['\"]?\s*$",
        s, re.IGNORECASE,
    ):
        return True
    # "Express your answer.", "Express the value of u."
    if re.match(r"^(?:Express|Represent)\s+", s, re.IGNORECASE):
        if len(s.split()) <= 5:
            return True
    return False


def _clean_sentence(sentence: str) -> str | None:
    """Clean a single sentence. Returns None if it should be removed entirely."""
    s = sentence.strip()
    if not s:
        return None

    # Remove standalone "Denote the X as 'Y'" sentences
    if _DENOTE_SENTENCE_RE.match(s):
        return None

    # Remove standalone "Represent the area as 'x'" sentences (no "in terms of")
    if _REPRESENT_SENTENCE_RE.match(s):
        return None

    # Strip trailing "in terms of / in relation to" phrases
    s = _strip_in_terms_of(s)

    # Strip trailing ", denoted as/by ..." phrases
    s = _strip_denoted_trailing(s)

    # Remove if the sentence became meaningless
    if _is_meaningless_after_strip(s):
        return None

    return s if s else None


def clean_variable_phrases(question: str) -> str:
    """
    Clean synthetic variable phrases from question text.

    1. Strip trailing "in terms of [X]" / "in relation to [X]" (unless X is π).
    2. Strip trailing ", denoted as/by ..." / ", represented as/by ..." phrases.
    3. Remove standalone "Denote..." / "Represent the ... as 'X'" sentences.
    4. Remove sentences that became meaningless after stripping.
    """
    lines = question.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue

        # Split line into sentences so we can remove trailing synthetic sentences
        # e.g. "... in terms of π. Represent the area as 'A'."
        parts = re.split(r"(?<=[.?!])\s+", stripped)
        kept_parts = []
        for part in parts:
            cleaned = _clean_sentence(part)
            if cleaned:
                kept_parts.append(cleaned)

        if kept_parts:
            cleaned_lines.append(" ".join(kept_parts))
    return "\n".join(cleaned_lines).strip()


# ── Main processing ──────────────────────────────────────────────────────────

def process_example(ex: dict) -> tuple[dict, dict]:
    """
    Convert one MCQ example to open-ended and normalise the answer.

    Returns (new_example, changes_dict).
    """
    question = ex.get("question", "")
    gt_letter = str(ex.get("gt", "")).strip().upper()
    changes = {}

    # 1) Parse choices & extract raw answer
    choices, choices_match = parse_choices(question)
    raw_answer = None
    if choices and gt_letter in choices:
        raw_answer = choices[gt_letter]
        changes["raw_answer"] = raw_answer
        changes["choices_found"] = choices

    # 2) Remove choices block
    if choices_match:
        before = question[: choices_match.start()]
        after = question[choices_match.end() :]
        question = (before + "\n" + after) if (before.strip() and after.strip()) else (before + after)
        changes["removed_choices"] = True

    # 3) Remove MCQ instructions
    question_cleaned = remove_instructions(question)
    if question_cleaned != question.strip():
        changes["removed_instructions"] = True
    question = question_cleaned

    # 4) Clean synthetic variable phrases
    question_cleaned = clean_variable_phrases(question)
    if question_cleaned != question:
        changes["cleaned_variable_phrases"] = True
    question = question_cleaned

    # 5) Normalise the extracted answer
    normalized_answer = None
    if raw_answer is not None:
        normalized_answer = normalize_geoqa_ground_truth(raw_answer)
        if normalized_answer != raw_answer:
            changes["answer_normalized"] = True

    # Build output
    new_ex = dict(ex)
    new_ex["question"] = question
    if normalized_answer is not None:
        new_ex["extracted_answer"] = normalized_answer
    elif raw_answer is not None:
        # Normalization returned empty (shouldn't happen), keep raw
        new_ex["extracted_answer"] = raw_answer

    return new_ex, changes


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convert geoqa MCQ → open questions + normalise answers"
    )
    ap.add_argument(
        "--input",
        default="/workspace/vlm-grpo-qwen25vl3b/data/vlaa_thinking_raw/geoqa/geoqa_only.jsonl",
    )
    ap.add_argument("--output_jsonl", default=None)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    input_path = Path(args.input)
    default_dir = Path("/workspace/vlm-grpo-qwen25vl3b/data/vlaa_thinking_raw/geoqa")
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else default_dir / "geoqa_open_normalized.jsonl"
    output_json = Path(args.output_json) if args.output_json else default_dir / "geoqa_open_normalized_preview.json"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:        {input_path}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Output JSON:  {output_json}")

    # Counters
    total = 0
    kept = 0
    no_choices = 0
    no_answer = 0
    instructions_removed = 0
    answers_normalized = 0
    skip_counts: dict[str, int] = {}
    all_preview = []

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_jsonl.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            total += 1

            # ── Filter ─────────────────────────────────────────────────
            skip_reason = should_skip(ex)
            if skip_reason:
                skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
                continue

            original_q = ex.get("question", "")
            new_ex, changes = process_example(ex)
            fout.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
            kept += 1

            if "choices_found" not in changes:
                no_choices += 1
            if "raw_answer" not in changes:
                no_answer += 1
            if changes.get("removed_instructions"):
                instructions_removed += 1
            if changes.get("answer_normalized"):
                answers_normalized += 1

            # Preview record (skip large fields)
            preview = {
                "id": ex.get("id", ""),
                "question_before": original_q[:200],
                "question_after": new_ex["question"][:200],
                "gt_letter": str(ex.get("gt", "")),
                "raw_answer": changes.get("raw_answer"),
                "extracted_answer": new_ex.get("extracted_answer"),
                "answer_normalized": changes.get("answer_normalized", False),
            }
            all_preview.append(preview)

    # Write preview JSON
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(all_preview, f, ensure_ascii=False, indent=2)

    # Summary
    total_skipped = sum(skip_counts.values())
    print(f"\nRead {total} examples, kept {kept}, skipped {total_skipped}.")
    if skip_counts:
        print(f"  Skip reasons:")
        for reason, count in sorted(skip_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25}: {count}")
    print(f"  No choices found:      {no_choices}")
    print(f"  No answer extracted:   {no_answer}")
    print(f"  Instructions removed:  {instructions_removed}")
    print(f"  Answers normalized:    {answers_normalized} / {kept - no_answer}")
    print(f"\nOutput JSONL → {output_jsonl}")
    print(f"Preview JSON → {output_json}")

    # Show some normalization examples
    norm_examples = [p for p in all_preview if p["answer_normalized"]]
    if norm_examples:
        print(f"\nSample normalizations ({min(20, len(norm_examples))} of {len(norm_examples)}):")
        for p in norm_examples[:20]:
            print(f"  {p['raw_answer']!r:35s} → {p['extracted_answer']}")


if __name__ == "__main__":
    main()
