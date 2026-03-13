#!/usr/bin/env python3
"""Simplify SFT response format: remove <facts>/<theorems>/<reasoning> tags,
remove [Fn]/[Tn] citations, convert inline functional annotations to natural
language, and produce a clean <think>Step 1:...Step 2:...</think><answer>...</answer>.

Usage:
    python scripts/PGPS9K/3c_simplify_sft_responses.py
    python scripts/PGPS9K/3c_simplify_sft_responses.py --dry_run   # preview only
"""

import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Functional annotation → natural language  (for inline occurrences)
# ---------------------------------------------------------------------------

def _parse_nested(s: str) -> list[str]:
    """Split top-level comma-separated args, respecting nested parens."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _convert_inner(expr: str) -> str:
    """Recursively convert a functional expression to natural language."""
    expr = expr.strip()
    m = re.match(r'^(\w+)\((.+)\)$', expr, re.DOTALL)
    if not m:
        return expr

    pred = m.group(1)
    args = _parse_nested(m.group(2).strip())
    conv = [_convert_inner(a) for a in args]

    if pred == "Line" and len(conv) == 2:
        return f"line {conv[0]}{conv[1]}"
    if pred == "Collinear":
        if len(conv) == 2:
            return f"points {conv[0]} and {conv[1]} are collinear"
        if len(conv) >= 3:
            return "points " + ", ".join(conv[:-1]) + f", and {conv[-1]} are collinear"
    if pred == "Length":
        if len(conv) == 2:
            return f"the length of segment {conv[0]}{conv[1]}"
        if len(conv) == 1:
            # Single-arg form: Length(AB) → AB (used inline in math)
            return conv[0]
    if pred == "Angle":
        if len(conv) == 3:
            return f"angle {conv[0]}{conv[1]}{conv[2]}"
        if len(conv) == 2:
            return f"angle {conv[0]}{conv[1]}"
        if len(conv) == 1:
            # Single-arg form: Angle(ABC) → angle ABC
            return f"angle {conv[0]}"
    if pred == "Perpendicular":
        if len(conv) == 3:
            return f"{conv[0]} is perpendicular to {conv[1]} at {conv[2]}"
        if len(conv) == 2:
            return f"{conv[0]} is perpendicular to {conv[1]}"
    if pred == "Parallel" and len(conv) == 2:
        return f"{conv[0]} is parallel to {conv[1]}"
    if pred == "Circle":
        if len(conv) >= 2:
            return f"circle with center {conv[0]} through {', '.join(conv[1:])}"
        return f"circle {conv[0]}"
    if pred == "Point" and len(conv) == 1:
        return f"point {conv[0]}"
    if pred == "Segment" and len(conv) == 2:
        return f"segment {conv[0]}{conv[1]}"
    if pred == "Triangle" and len(conv) == 3:
        return f"triangle {conv[0]}{conv[1]}{conv[2]}"
    if pred == "Quadrilateral" and len(conv) >= 4:
        return f"quadrilateral {''.join(conv)}"
    if pred == "Polygon" and len(conv) >= 3:
        return f"polygon {''.join(conv)}"
    if pred == "Midpoint" and len(conv) == 2:
        return f"{conv[0]} is the midpoint of {conv[1]}"
    if pred == "Bisects" and len(conv) == 2:
        return f"{conv[0]} bisects {conv[1]}"
    if pred == "Congruent" and len(conv) == 2:
        return f"{conv[0]} is congruent to {conv[1]}"
    if pred == "Similar" and len(conv) == 2:
        return f"{conv[0]} is similar to {conv[1]}"
    if pred == "Equal" and len(conv) == 2:
        return f"{conv[0]} equals {conv[1]}"
    if pred in ("ArcDegree", "Arc") and len(conv) == 2:
        return f"the measure of arc {conv[0]}{conv[1]}"
    if pred == "ArcLength" and len(conv) == 2:
        return f"the arc length from {conv[0]} to {conv[1]}"
    if pred == "MajorArcDegree" and len(conv) == 2:
        return f"the measure of major arc {conv[0]}{conv[1]}"
    if pred == "MajorArcLength" and len(conv) == 2:
        return f"the major arc length from {conv[0]} to {conv[1]}"
    if pred in ("ArcMeasure", "MeasureOfArc") and len(conv) == 3:
        return f"the measure of arc {conv[0]}{conv[1]}{conv[2]}"
    if pred == "Diameter" and len(conv) == 1:
        return f"the diameter of {conv[0]}"
    if pred == "Perimeter" and len(conv) == 1:
        return f"the perimeter of {conv[0]}"
    if pred == "IsRhombus":
        return f"quadrilateral {''.join(conv)} is a rhombus"
    if pred in ("Measure", "MeasureOf", "MeasureOfAngle"):
        if len(conv) == 1:
            return f"the measure of {conv[0]}"
        if len(conv) == 3:
            return f"the measure of angle {conv[0]}{conv[1]}{conv[2]}"
    if pred == "Circumcircle":
        return f"the circumscribed circle of {''.join(conv)}"

    # Fallback: keep as-is
    return f"{pred}({', '.join(conv)})"


# Known predicates that appear inline in reasoning text
_INLINE_PREDICATES = (
    "Length", "Angle", "Line", "Collinear", "Perpendicular", "Parallel",
    "Circle", "Point", "Segment", "Triangle", "Quadrilateral", "Polygon",
    "ArcDegree", "ArcLength", "Arc", "MajorArcDegree", "MajorArcLength",
    "ArcMeasure", "MeasureOfArc", "Midpoint", "Bisects", "Congruent",
    "Similar", "Equal", "Diameter", "Perimeter", "IsRhombus",
    "Measure", "MeasureOf", "MeasureOfAngle", "Circumcircle",
)


def _find_balanced_paren(text: str, start: int) -> int | None:
    """Find the index of the closing ')' matching the '(' at `start`."""
    if start >= len(text) or text[start] != '(':
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return None


def convert_inline_annotations(text: str) -> str:
    """Find and convert inline functional annotations like Length(A, B) in text."""
    # Build pattern to match known predicates followed by '('
    pred_pattern = '|'.join(_INLINE_PREDICATES)
    pattern = re.compile(rf'\b({pred_pattern})\(')

    result = []
    pos = 0
    while pos < len(text):
        m = pattern.search(text, pos)
        if not m:
            result.append(text[pos:])
            break

        # Add text before the match
        result.append(text[pos:m.start()])

        # Find balanced closing paren
        paren_start = m.start() + len(m.group(1))
        paren_end = _find_balanced_paren(text, paren_start)

        if paren_end is None:
            # Unbalanced — keep as-is and move on
            result.append(text[m.start():m.end()])
            pos = m.end()
            continue

        # Extract full expression and convert
        full_expr = text[m.start():paren_end + 1]
        converted = _convert_inner(full_expr)
        result.append(converted)
        pos = paren_end + 1

    return "".join(result)


# ---------------------------------------------------------------------------
# Parse response blocks
# ---------------------------------------------------------------------------

def parse_facts(response: str) -> dict[str, str]:
    """Extract {label: raw_text} from <facts> block."""
    m = re.search(r'<facts>(.*?)</facts>', response, re.DOTALL)
    if not m:
        return {}
    facts = {}
    for line in m.group(1).strip().split('\n'):
        line = line.strip()
        fm = re.match(r'\[(F\d+)\]\s*(.*)', line)
        if fm:
            facts[fm.group(1)] = fm.group(2).strip()
    return facts


def parse_theorems(response: str) -> dict[str, str]:
    """Extract {label: text} from <theorems> block."""
    m = re.search(r'<theorems>(.*?)</theorems>', response, re.DOTALL)
    if not m:
        return {}
    theorems = {}
    for line in m.group(1).strip().split('\n'):
        line = line.strip()
        tm = re.match(r'\[(T\d+)\]\s*(.*)', line)
        if tm:
            theorems[tm.group(1)] = tm.group(2).strip()
    return theorems


def parse_reasoning(response: str) -> str:
    m = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_answer(response: str) -> str:
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    return m.group(1).strip() if m else ""


def _fact_to_nl(raw: str) -> str:
    """Convert a single raw fact to natural language."""
    raw = raw.strip()
    # Handle equality: Predicate(args) = value
    eq_match = re.match(r'^(.+?)\s*=\s*(.+)$', raw)
    if eq_match:
        lhs_raw = eq_match.group(1).strip()
        rhs_raw = eq_match.group(2).strip()
        lhs = _convert_inner(lhs_raw) if re.match(r'^\w+\(', lhs_raw) else lhs_raw
        rhs = _convert_inner(rhs_raw) if re.match(r'^\w+\(', rhs_raw) else rhs_raw
        return f"{lhs} = {rhs}"
    if re.match(r'^\w+\(', raw):
        return _convert_inner(raw)
    return raw


def _theorem_name(text: str) -> str:
    """Extract a short theorem name from the full theorem text."""
    if ':' in text:
        return text.split(':')[0].strip()
    if len(text) > 80:
        return text[:77] + "..."
    return text


# ---------------------------------------------------------------------------
# Inline [Fn] / [Tn] citations with actual content
# ---------------------------------------------------------------------------

def inline_citations(text: str, facts_nl: dict[str, str],
                     theorems: dict[str, str]) -> str:
    """Replace [Fn] and [Tn] citation markers with actual content.

    - [Fn] → the natural language fact
    - [Tn] after theorem name already present → remove tag
    - [Tn] as standalone reference → inline theorem name
    """

    def replace_fact(match: re.Match) -> str:
        label = match.group(1)  # e.g. 'F1'
        nl = facts_nl.get(label)
        return nl if nl else match.group(0)

    def replace_theorem(match: re.Match) -> str:
        label = match.group(1)  # e.g. 'T1'
        text = theorems.get(label)
        return _theorem_name(text) if text else match.group(0)

    # Step 1: Clean up "From the visual fact [Fn]" → "From the fact that <fact>"
    def _replace_from_fact(m: re.Match) -> str:
        labels = re.findall(r'\[(F\d+)\]', m.group(0))
        parts = [facts_nl.get(l, l) for l in labels]
        joined = " and ".join(parts)
        return f"Since {joined}"

    text = re.sub(
        r'[Ff]rom\s+(?:the\s+)?(?:visual\s+)?facts?\s+\[F\d+\](?:\s*(?:,\s*|\s+and\s+)\[F\d+\])*',
        _replace_from_fact,
        text,
    )

    # Step 2: "From [Fn]" / "From [Fn] and [Fn]" → "Since <fact> and <fact>"
    def _replace_from_ref(m: re.Match) -> str:
        labels = re.findall(r'\[(F\d+)\]', m.group(0))
        parts = [facts_nl.get(l, l) for l in labels]
        joined = " and ".join(parts)
        return f"Since {joined}"

    text = re.sub(
        r'[Ff]rom\s+\[F\d+\](?:\s*(?:,\s*|\s+and\s+)\[F\d+\])*',
        _replace_from_ref,
        text,
    )

    # Step 3: "facts [Fn] and [Fn]" → "the facts that <fact> and <fact>"
    def _replace_facts_ref(m: re.Match) -> str:
        labels = re.findall(r'\[(F\d+)\]', m.group(0))
        parts = [facts_nl.get(l, l) for l in labels]
        joined = " and ".join(parts)
        return f"the facts that {joined}"

    text = re.sub(
        r'facts?\s+\[F\d+\](?:\s*(?:,\s*|\s+and\s+)\[F\d+\])*',
        _replace_facts_ref,
        text,
    )

    # Step 4: "in [Fn] that" → "that <fact>, that"
    #         "in [Fn]," → "(i.e., <fact>),"
    def _replace_in_fact(m: re.Match) -> str:
        label = re.search(r'\[(F\d+)\]', m.group(0)).group(1)
        nl = facts_nl.get(label, '')
        trailing = m.group(0).rstrip()
        if trailing.endswith('that'):
            return f" that {nl}, that"
        return f" ({nl})"

    text = re.sub(r'\s+in\s+\[F\d+\](?:\s+that)?', _replace_in_fact, text)

    # Step 5: Theorem handling — context-aware replacement
    # 5a: "according to [Tn]" / "According to the [Tn]" → inline theorem name
    def _replace_theorem_ref(m: re.Match) -> str:
        label = re.search(r'\[(T\d+)\]', m.group(0)).group(1)
        name = _theorem_name(theorems.get(label, ''))
        prefix = m.group(0)[:m.group(0).index('[')].rstrip()
        return f"{prefix} {name}" if name else prefix

    text = re.sub(
        r'(?:[Aa]ccording to|[Ss]tated in|[Gg]iven in|[Bb]y|[Aa]pplying)\s+(?:the\s+)?\[T\d+\]',
        _replace_theorem_ref,
        text,
    )

    # 5b: Remaining [Tn] (typically "TheoremName [Tn]") → just remove tag
    text = re.sub(r'\s*\[T\d+\]', '', text)

    # Step 6: Remaining standalone [Fn] — inline the fact
    text = re.sub(r'\[F\d+\]', lambda m: replace_fact(re.match(r'\[(F\d+)\]', m.group(0))), text)

    return text


# ---------------------------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------------------------

def cleanup(text: str) -> str:
    """Clean up artifacts from citation inlining and annotation conversion."""
    # Remove backticks (some Gemini responses wrap annotations in backticks)
    text = text.replace('`', '')

    # "we know that that" (double that from inlining)
    text = re.sub(r'\bthat that\b', 'that', text)

    # ",." or ", ." → "."
    text = re.sub(r',\s*\.', '.', text)

    # ", ," or ",,"
    text = re.sub(r',\s*,', ',', text)

    # Double spaces
    text = re.sub(r'  +', ' ', text)

    # Space before punctuation
    text = re.sub(r' ([,.])', r'\1', text)

    # Leading/trailing whitespace per line
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text


# ---------------------------------------------------------------------------
# Transform a single response
# ---------------------------------------------------------------------------

def transform_response(response: str) -> str | None:
    """Transform a structured response to simplified format."""
    facts_raw = parse_facts(response)
    theorems = parse_theorems(response)
    reasoning = parse_reasoning(response)
    answer = parse_answer(response)

    if not reasoning or not answer:
        return None

    # Convert facts to natural language
    facts_nl = {label: _fact_to_nl(raw) for label, raw in facts_raw.items()}

    # Step 1: Inline [Fn] and [Tn] citations with actual content
    text = inline_citations(reasoning, facts_nl, theorems)

    # Step 2: Convert remaining inline functional annotations to natural language
    text = convert_inline_annotations(text)

    # Step 3: Clean up
    text = cleanup(text)

    return f"<think>\n{text}\n</think>\n<answer>{answer}</answer>"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path, dry_run: bool = False):
    """Process a JSONL file of SFT responses."""
    rows = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} rows from {input_path.name}")

    transformed = []
    failed = 0
    for row in rows:
        new_response = transform_response(row["response"])
        if new_response is None:
            failed += 1
            continue
        new_row = dict(row)
        new_row["response"] = new_response
        transformed.append(new_row)

    print(f"  Transformed: {len(transformed)}, Failed to parse: {failed}")

    # Show examples
    for i, row in enumerate(transformed[:5]):
        print(f"\n{'='*70}")
        print(f"Example {i+1}: {row['id']} ({row['type']})")
        print(f"{'='*70}")
        print(row["response"][:1200])
        if len(row["response"]) > 1200:
            print("...")

    # Check for any remaining functional annotations or citations
    remaining_citations = 0
    remaining_annotations = 0
    for row in transformed:
        resp = row["response"]
        if re.search(r'\[(?:F|T)\d+\]', resp):
            remaining_citations += 1
        if re.search(r'\b(?:Length|Angle|Collinear|Perpendicular|Parallel)\([A-Z]', resp):
            remaining_annotations += 1

    print(f"\n--- Quality check ---")
    print(f"Remaining [Fn]/[Tn] citations: {remaining_citations}/{len(transformed)}")
    print(f"Remaining inline annotations: {remaining_annotations}/{len(transformed)}")

    # Token length stats
    lengths = [len(r["response"]) for r in transformed]
    lengths.sort()
    n = len(lengths)
    print(f"Response char lengths: min={lengths[0]}, median={lengths[n//2]}, "
          f"P95={lengths[int(n*0.95)]}, max={lengths[-1]}")

    if not dry_run:
        with open(output_path, "w") as f:
            for row in transformed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWritten to {output_path}")

    return transformed


def main():
    parser = argparse.ArgumentParser(description="Simplify SFT response format")
    parser.add_argument("--dry_run", action="store_true", help="Preview only, don't write")
    args = parser.parse_args()

    files = [
        (BASE / "data/PGPS9K/sft_responses_train.jsonl",
         BASE / "data/PGPS9K/sft_responses_train_simplified.jsonl"),
        (BASE / "data/PGPS9K/sft_responses_val.jsonl",
         BASE / "data/PGPS9K/sft_responses_val_simplified.jsonl"),
    ]

    for input_path, output_path in files:
        if input_path.exists():
            print(f"\n{'#'*70}")
            print(f"Processing {input_path.name}")
            print(f"{'#'*70}")
            process_file(input_path, output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
