#!/usr/bin/env python3
"""
Normalise the "gt" field in synthesis_only.jsonl.

Three transformations:
1. Remove "Output", "Output 4", "Output:", "Output 4:" prefixes.
2. When gt is "a = <value>" (without a full y= equation), look up the
   equation template "y = ... a" in meta.text_question, substitute
   the value of a, and use the resulting "y = ..." as the new gt.
3. When gt contains both "y = ..." and "a = ...", extract just the
   "y = ..." equation and discard the "a = ..." part.

Usage:
    python scripts/synthesis/normalize_synthesis_gt.py
    python scripts/synthesis/normalize_synthesis_gt.py --input <path> --output <path>
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.extract_answer import normalize_answer_comprehensive


# ── 1. Strip "Output ..." prefix ─────────────────────────────────────────────

_OUTPUT_PREFIX_RE = re.compile(
    r"^\s*Output(?:\s+\d+)?\s*:?\s*\n?",
    re.IGNORECASE,
)


def strip_output_prefix(gt: str) -> str:
    """Remove leading 'Output', 'Output 4', 'Output:', 'Output 4:' etc."""
    return _OUTPUT_PREFIX_RE.sub("", gt)


# ── 2. Extract a-value from gt and substitute into equation ──────────────────

# Match  \boxed{a = <expr>}  or bare  a = <expr>  (possibly with \[ \] wrapper)
_A_VALUE_RE = re.compile(
    r"a\s*=\s*(.+)",
    re.DOTALL,
)


def _extract_a_value(gt_clean: str) -> str | None:
    """
    If *gt_clean* is essentially "a = <value>" (no y= present),
    return the value part.  Otherwise return None.
    """
    if "y =" in gt_clean or "y=" in gt_clean:
        return None

    m = _A_VALUE_RE.search(gt_clean)
    if not m:
        return None

    val = m.group(1).strip()
    # Strip trailing LaTeX delimiters / braces
    val = val.rstrip("} \\]$.")
    # Remove \quad and \text{...} annotations
    val = re.sub(r"\\quad.*", "", val).strip()
    val = re.sub(r"\\text\{[^}]*\}", "", val).strip()
    return val if val else None


# Match "y = ... a ..." in text_question
_EQUATION_RE = re.compile(
    r"(y\s*=\s*[^,;]*?\ba\b[^,;]*?)(?:\s+(?:for|that|passing|through|intersect|given|of|the)|[,.]|\s*$)",
    re.IGNORECASE,
)


def _find_equation_template(text_question: str) -> str | None:
    """Extract the 'y = ... a ...' equation template from text_question."""
    m = _EQUATION_RE.search(text_question)
    if m:
        return m.group(1).strip()
    # Fallback: grab everything from 'y =' up to the next clause
    m2 = re.search(r"(y\s*=\s*[^.?]+)", text_question, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


def _substitute_a(equation: str, a_value: str) -> str:
    """
    Replace the parameter 'a' in *equation* with *a_value*.

    Handles patterns like:
      - "x + a"  → "x + <val>"
      - "a * sin" → "<val> * sin"
      - "(x + a)" → "(x + <val>)"
    """
    # Wrap in parens if a_value has operators (e.g. "pi/2", "-5")
    needs_parens = any(c in a_value for c in "+-/") and not a_value.startswith("(")

    # Use word-boundary replacement so we don't clobber 'tan', 'abs', etc.
    def _repl(m: re.Match) -> str:
        if needs_parens:
            return f"({a_value})"
        return a_value

    return re.sub(r"\ba\b", _repl, equation)


def resolve_a_to_equation(gt_clean: str, meta: dict) -> str | None:
    """
    If gt is 'a = <value>', find the equation in text_question,
    substitute a, and return the full 'y = ...' string.
    Returns None if not applicable.
    """
    a_val = _extract_a_value(gt_clean)
    if a_val is None:
        return None

    text_question = meta.get("text_question", "")
    if not text_question:
        return None

    equation = _find_equation_template(text_question)
    if equation is None:
        return None

    return _substitute_a(equation, a_val)


# ── 3. Extract y= equation when both y= and a= are present ──────────────────

def _extract_boxed_content(text: str, start: int) -> str | None:
    """Extract content from \\boxed{...} at *start*, handling nested braces."""
    prefix = "\\boxed{"
    if text[start:start + len(prefix)] != prefix:
        return None
    i = start + len(prefix)
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start + len(prefix):i - 1]
    return None


def _extract_y_equation(gt: str) -> str | None:
    """
    When gt contains both 'y =' and 'a =', extract just the y= expression.
    Returns the y= equation string, or None if not applicable.
    """
    if "a =" not in gt and "a=" not in gt:
        return None
    if "y =" not in gt and "y=" not in gt:
        return None

    # Strategy 1: find \boxed{y = ...} with proper brace matching
    for m in re.finditer(r"\\boxed\{", gt):
        content = _extract_boxed_content(gt, m.start())
        if content and re.match(r"\s*y\s*=", content):
            # Clean trailing \quad \text{...} annotations
            expr = re.sub(r"\s*\\quad.*", "", content).strip()
            expr = re.sub(r"\s*\\text\{[^}]*\}\s*", "", expr).strip()
            return expr

    # Strategy 2: find y = ... inside \text{} or other context
    # Use a balanced approach: grab from "y =" to end, then trim a= parts
    m = re.search(r"(y\s*=\s*.+)", gt)
    if m:
        expr = m.group(1)
        # Remove everything from "a =" or "\boxed{a" onward
        expr = re.split(r"[,;]\s*\\?boxed\{a\s*=|,\s*a\s*=|\}\s*with\s|when\s", expr)[0]
        # Clean trivial \text{...} artifacts before stripping (e.g. \text{)}, \text{})
        expr = re.sub(r"\s*\\text\{[^a-zA-Z0-9]*\}", "", expr).strip()
        # Strip trailing LaTeX noise
        expr = expr.rstrip(" \\]$)}")
        # Clean any leftover broken \text{ remnants
        expr = re.sub(r"\s*\\text\{\s*$", "", expr).strip()
        # Re-balance: ensure parens/braces are matched
        expr = _rebalance(expr)
        return expr

    return None


def _rebalance(expr: str) -> str:
    """Append missing closing parens/braces in correct nesting order."""
    # Clean empty \text{} artifacts first
    expr = re.sub(r"\s*\\text\{\s*\}", "", expr).strip()

    # Track open/close stack to emit closers in correct order
    stack: list[str] = []
    closer = {"(": ")", "{": "}"}
    for ch in expr:
        if ch in closer:
            stack.append(closer[ch])
        elif ch in (")", "}"):
            # Pop matching opener if present
            if stack and stack[-1] == ch:
                stack.pop()
    # Append any unclosed brackets in reverse (innermost first)
    if stack:
        expr += "".join(reversed(stack))
    return expr


# ── 4. Unwrap LaTeX wrappers (\[\boxed{...}\], \[...\]) ─────────────────────

def _unwrap_latex(gt: str) -> str:
    """Remove outer \\[...\\] and \\boxed{...} wrappers, keeping inner content."""
    s = gt.strip()
    # Strip \[...\]
    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    # Strip \boxed{...}
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[7:-1].strip()
    return s


# ── Main ─────────────────────────────────────────────────────────────────────

def normalise_gt(row: dict) -> tuple[str, str]:
    """
    Normalise the gt field of a single row.

    Returns (new_gt, method) where method describes what was done.
    """
    gt_raw = row.get("gt", "")

    # Step 1: strip Output prefix
    gt = strip_output_prefix(gt_raw)

    # Step 2: unwrap LaTeX for inspection
    gt_inner = _unwrap_latex(gt)

    # Step 3: if gt is "a = ...", resolve to full equation
    meta = row.get("meta", {})
    if isinstance(meta, str):
        try:
            meta = ast.literal_eval(meta)
        except Exception:
            meta = {}

    resolved = resolve_a_to_equation(gt_inner, meta)
    if resolved is not None:
        return f"\\[\\boxed{{{resolved}}}\\]", "a_resolved"

    # Step 4: if gt has both "y = ..." and "a = ...", keep only the y= part
    y_eq = _extract_y_equation(gt)
    if y_eq is not None:
        return f"\\[\\boxed{{{y_eq}}}\\]", "y_extracted"

    # If we only stripped the Output prefix, re-wrap what remains
    if gt != gt_raw:
        return gt.strip(), "output_stripped"

    return gt_raw, "unchanged"


def main():
    ap = argparse.ArgumentParser(
        description="Normalise gt field in synthesis_only.jsonl"
    )
    ap.add_argument(
        "--input",
        default="/workspace/vlm-grpo-qwen25vl3b/data/vlaa_thinking_raw/synthesis/synthesis_only.jsonl",
    )
    ap.add_argument(
        "--output",
        default="/workspace/vlm-grpo-qwen25vl3b/data/vlaa_thinking_raw/synthesis/synthesis_normalized.jsonl",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    total = 0
    method_counts: dict[str, int] = {}
    samples: dict[str, list] = {}

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1

            gt_raw = row.get("gt", "")
            new_gt, method = normalise_gt(row)
            method_counts[method] = method_counts.get(method, 0) + 1

            row["gt"] = new_gt

            # Normalise inner math content via normalize_answer_comprehensive
            inner = _unwrap_latex(new_gt)
            # Also unwrap nested \boxed{} that _unwrap_latex may leave
            # (e.g. when gt was only partially wrapped)
            bm = re.match(r"^\\boxed\{(.+)\}$", inner, re.DOTALL)
            if bm:
                inner = bm.group(1).strip()
            normalized = normalize_answer_comprehensive(inner) if inner else inner
            row["extracted_answer"] = normalized

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            if method != "unchanged" or normalized != f"${inner}$":
                samples.setdefault(method, []).append({
                    "id": row.get("id", ""),
                    "gt_raw": gt_raw[:150],
                    "gt_new": new_gt[:150],
                    "extracted": normalized[:150],
                })

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nProcessed {total} examples.")
    print(f"\nMethod distribution:")
    for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
        pct = 100 * c / total if total > 0 else 0
        print(f"  {m:<20}: {c:>5} ({pct:.1f}%)")

    print(f"\n{'─' * 90}")
    print("Samples per method:")
    for method in sorted(samples):
        entries = samples[method][:8]
        print(f"\n  [{method}] ({len(samples[method])} total)")
        for s in entries:
            print(f"    {s['id']}")
            print(f"      before:    {s['gt_raw']!r}")
            print(f"      after:     {s['gt_new']!r}")
            print(f"      extracted: {s['extracted']!r}")

    print(f"\n{'─' * 90}")
    print(f"Output → {output_path}")


if __name__ == "__main__":
    main()
