#!/usr/bin/env python3
"""Generate SFT reasoning responses for PGPS9K using Google Gemini 2.5 Pro.

Sends geometry problems (image + facts + question) to Gemini and collects
step-by-step reasoning in <think><facts><theorem><reasoning></think><answer> format.

Supports multiple API keys for round-robin rotation to multiply daily quotas:
    export GOOGLE_API_KEYS=key1,key2,key3

Or a single key:
    export GOOGLE_API_KEY=<key>

Usage:
    python scripts/PGPS9K/2_generate_sft_responses.py --limit 3          # test run
    python scripts/PGPS9K/2_generate_sft_responses.py --samples 4        # rejection sampling
    python scripts/PGPS9K/2_generate_sft_responses.py                    # full run
    python scripts/PGPS9K/2_generate_sft_responses.py --no-resume        # start fresh
"""

import argparse
import asyncio
import base64
import itertools
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit(
        "google-genai package not found. Install with:\n"
        "  pip install google-genai"
    )

# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------
def load_api_keys() -> list[str]:
    """Load API keys from GOOGLE_API_KEYS (comma-separated) or GOOGLE_API_KEY."""
    keys_str = os.environ.get("GOOGLE_API_KEYS", "")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GOOGLE_API_KEY", "")
    if single.strip():
        return [single.strip()]
    sys.exit(
        "No API key found. Set GOOGLE_API_KEYS=key1,key2,... or GOOGLE_API_KEY=<key>"
    )


def create_client_pool(api_keys: list[str]) -> list[genai.Client]:
    """Create a Gemini client for each API key."""
    return [genai.Client(api_key=k) for k in api_keys]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/workspace/vlm-grpo-qwen25vl3b")
DEFAULT_INPUT = BASE / "data/PGPS9K/sft_prompts_train.jsonl"
DEFAULT_OUTPUT = BASE / "data/PGPS9K/sft_responses_train.jsonl"

# ---------------------------------------------------------------------------
# Required tags for validation
# ---------------------------------------------------------------------------
REQUIRED_TAGS = ["<think>", "<facts>", "<theorems>", "<reasoning>", "<answer>"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_response(text: str) -> bool:
    """Check that the response contains all required XML tags and citations."""
    if not all(tag in text for tag in REQUIRED_TAGS):
        return False
    # Check [F] and [T] labels exist in <facts> and <theorems>
    facts_m = re.search(r"<facts>(.*?)</facts>", text, re.DOTALL)
    theorems_m = re.search(r"<theorems>(.*?)</theorems>", text, re.DOTALL)
    reasoning_m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if not (facts_m and theorems_m and reasoning_m):
        return False
    if not re.search(r"\[F\d+\]", facts_m.group(1)):
        return False
    if not re.search(r"\[T\d+\]", theorems_m.group(1)):
        return False
    # Reasoning must cite both [F] and [T] labels and use Step numbering
    reasoning = reasoning_m.group(1)
    if not re.search(r"\[F\d+\]", reasoning):
        return False
    if not re.search(r"\[T\d+\]", reasoning):
        return False
    if not re.search(r"Step\s+\d+", reasoning):
        return False
    # Answer must use LaTeX, not plain-text math
    answer_m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_m:
        ans = answer_m.group(1)
        # Reject plain-text trig/sqrt: sin(, cos(, tan(, arcsin(, sqrt( without backslash
        if re.search(r"(?<!\\)(?:arc)?(?:sin|cos|tan)\(", ans):
            return False
        if re.search(r"(?<!\\)sqrt\(", ans):
            return False
    return True


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image_bytes(image_path: str) -> tuple[bytes, str] | None:
    """Load image file and return (raw_bytes, mime_type), or None."""
    p = Path(image_path)
    if not p.exists():
        return None
    suffix = p.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return p.read_bytes(), mime


# ---------------------------------------------------------------------------
# Process one entry
# ---------------------------------------------------------------------------
def check_answer_correct(entry: dict, response_text: str) -> bool:
    """Check if the response's answer matches the gold answer."""
    from utils.compare_answer import parse_numeric, parse_all_numeric, check_answer, extract_answer
    gold_val = parse_numeric(str(entry.get("answer", "")))
    if gold_val is None:
        return False
    pred_raw = extract_answer(response_text) or ""
    pred_vals = parse_all_numeric(pred_raw)
    if not pred_vals:
        return False
    return check_answer(pred_vals, gold_val)


async def process_entry(
    client: genai.Client,
    entry: dict,
    model: str,
    semaphore: asyncio.Semaphore,
    max_output_tokens: int = 4096,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> dict:
    """Send one problem to Gemini and collect the reasoning response."""
    # Extract system prompt and user content from messages
    messages = entry["messages"]
    system_text = messages[0]["content"]
    user_content = messages[1]["content"]

    # Build user text from the content blocks
    user_text = ""
    for block in user_content:
        if isinstance(block, dict) and block.get("type") == "text":
            user_text = block["text"]
            break

    # Load image
    image_path = entry.get("image", "")
    img_data = load_image_bytes(image_path) if image_path else None

    # Build contents for Gemini
    parts = []
    if img_data:
        raw_bytes, mime = img_data
        parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
    parts.append(types.Part.from_text(text=user_text))

    contents = [types.Content(role="user", parts=parts)]

    config = types.GenerateContentConfig(
        system_instruction=system_text,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF",
            ),
        ],
    )

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            # Extract text from response — response.text can return None
            # or raise on incomplete responses (MAX_TOKENS), so fall back
            # to extracting from candidates directly
            text = ""
            try:
                text = response.text or ""
            except Exception:
                pass
            if not text and response.candidates:
                try:
                    parts = response.candidates[0].content.parts
                    text = "".join(p.text for p in parts if hasattr(p, "text"))
                except Exception:
                    pass
            valid = validate_response(text)

            if valid:
                # Check answer correctness — don't retry wrong answers,
                # Gemini gives the same wrong answer consistently
                if check_answer_correct(entry, text):
                    return {
                        "id": entry["id"],
                        "image": entry.get("image", ""),
                        "type": entry.get("type", ""),
                        "answer": entry.get("answer", ""),
                        "prompt": user_text,
                        "response": text,
                        "valid_format": True,
                    }
                else:
                    print(
                        f"\n[WRONG] {entry['id']}: valid format but wrong answer, skipping retries",
                        file=sys.stderr,
                    )
                    return {
                        "id": entry["id"],
                        "image": entry.get("image", ""),
                        "type": entry.get("type", ""),
                        "answer": entry.get("answer", ""),
                        "prompt": user_text,
                        "response": text,
                        "valid_format": True,
                        "_wrong_answer": True,
                    }

            # Invalid response — retry with backoff
            if not text.strip():
                # Log why the response is empty
                finish_reason = None
                block_reason = None
                safety_ratings = None
                if response.candidates:
                    c = response.candidates[0]
                    finish_reason = getattr(c, "finish_reason", None)
                    safety_ratings = getattr(c, "safety_ratings", None)
                if hasattr(response, "prompt_feedback"):
                    pf = response.prompt_feedback
                    block_reason = getattr(pf, "block_reason", None)
                print(
                    f"\n[WARN] {entry['id']}: empty response (attempt {attempt+1}/{max_retries})"
                    f" finish_reason={finish_reason} block_reason={block_reason}"
                    f" safety={safety_ratings}",
                    file=sys.stderr,
                )
            else:
                missing = [t for t in REQUIRED_TAGS if t not in text]
                print(
                    f"\n[WARN] {entry['id']}: invalid format (attempt {attempt+1}/{max_retries}): {missing}",
                    file=sys.stderr,
                )

            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                continue

            # Final attempt failed — return what we have
            return {
                "id": entry["id"],
                "image": entry.get("image", ""),
                "type": entry.get("type", ""),
                "answer": entry.get("answer", ""),
                "prompt": user_text,
                "response": text,
                "valid_format": False,
            }

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate" in err_str or "resource" in err_str
            is_retryable = is_rate_limit or "500" in err_str or "503" in err_str

            if attempt == max_retries - 1 or not is_retryable:
                print(f"\n[ERROR] {entry['id']}: {e}", file=sys.stderr)
                return {
                    "id": entry["id"],
                    "image": entry.get("image", ""),
                    "type": entry.get("type", ""),
                    "answer": entry.get("answer", ""),
                    "prompt": user_text,
                    "response": "",
                    "valid_format": False,
                    "_error": str(e),
                }

            wait = min(2**attempt * 2, 120) + random.uniform(0, 2)
            if is_rate_limit:
                wait = max(wait, 10)  # at least 10s for rate limits
            await asyncio.sleep(wait)

    # Should not reach here, but just in case
    return {
        "id": entry["id"],
        "image": entry.get("image", ""),
        "type": entry.get("type", ""),
        "answer": entry.get("answer", ""),
        "prompt": user_text,
        "response": "",
        "valid_format": False,
        "_error": "max retries exceeded",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main(
    input_path: Path,
    output_path: Path,
    model: str,
    concurrency: int,
    max_output_tokens: int,
    temperature: float,
    resume: bool,
    limit: int,
    samples: int,
):
    # Load skip list (problems Gemini consistently gets wrong)
    skip_ids: set[str] = set()
    skip_file = BASE / "data/PGPS9K/gemini_2_5_pro_failed.jsonl"
    if skip_file.exists():
        with open(skip_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    skip_ids.add(json.loads(line)["id"])
        print(f"Skipping {len(skip_ids)} known-bad problem IDs from {skip_file}")

    # Load input
    entries = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry["id"] in skip_ids:
                    continue
                entries.append(entry)
    print(f"Loaded {len(entries)} entries from {input_path}")

    if limit > 0:
        entries = entries[:limit]
        print(f"Limited to first {limit} entries")

    # Expand entries for multi-sample generation
    if samples > 1:
        expanded = []
        for entry in entries:
            for s in range(samples):
                e = dict(entry)
                e["id"] = f"{entry['id']}_s{s}"
                expanded.append(e)
        entries = expanded
        print(f"Expanded to {len(entries)} entries ({samples} samples each)")

    # Resume: skip already-processed entries that are valid AND correct
    # Build a lookup of gold answers from input entries
    gold_answers = {e["id"]: str(e.get("answer", "")) for e in entries}

    done_ids: set[str] = set()
    if resume and output_path.exists():
        from utils.compare_answer import parse_numeric, parse_all_numeric, check_answer, extract_answer
        # Re-validate existing output: keep only good rows, move bad ones to failed file
        good_lines = []
        removed = 0
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "_error" in obj or not obj.get("valid_format", False):
                    removed += 1
                    continue
                gold_raw = obj.get("answer", "")
                pred_raw = extract_answer(obj.get("response", "")) or ""
                gold_val = parse_numeric(str(gold_raw))
                pred_vals = parse_all_numeric(pred_raw)
                if gold_val is not None and pred_vals and check_answer(pred_vals, gold_val):
                    good_lines.append(line)
                    done_ids.add(obj["id"])
                else:
                    removed += 1
                    # Move bad row to failed file
                    with open(skip_file, "a") as ff:
                        fail_entry = {
                            "id": obj["id"], "image": obj.get("image", ""),
                            "type": obj.get("type", ""), "answer": obj.get("answer", ""),
                        }
                        ff.write(json.dumps(fail_entry, ensure_ascii=False) + "\n")
        # Rewrite output file with only good rows
        with open(output_path, "w") as f:
            for line in good_lines:
                f.write(line + "\n")
        if removed:
            print(f"Cleaned {removed} bad rows from {output_path}")
        print(f"Resuming: {len(done_ids)} entries already done (valid + correct), skipping them")
        entries = [e for e in entries if e["id"] not in done_ids]
        if not entries:
            print("All entries already processed. Done.")
            return

    print(
        f"Processing {len(entries)} entries | model={model} | "
        f"concurrency={concurrency} | temperature={temperature} | "
        f"max_tokens={max_output_tokens}"
    )

    # Initialize Gemini clients (round-robin across API keys)
    api_keys = load_api_keys()
    clients = create_client_pool(api_keys)
    print(f"Using {len(clients)} API key(s)")
    client_cycle = itertools.cycle(clients)
    semaphore = asyncio.Semaphore(concurrency)

    pbar = tqdm(total=len(entries), desc="Generating responses")

    async def process_and_track(entry: dict, client: genai.Client) -> dict:
        result = await process_entry(
            client, entry, model, semaphore,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        pbar.update(1)
        return result

    tasks = [process_and_track(e, next(client_cycle)) for e in entries]

    # Write results incrementally — only good results to output,
    # wrong answers auto-appended to failed file
    mode = "a" if done_ids else "w"
    correct = 0
    wrong_answer = 0
    errors = 0
    invalid = 0
    with open(output_path, mode) as out_f, \
         open(skip_file, "a") as fail_f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if "_error" in result:
                errors += 1
            elif result.get("_wrong_answer"):
                wrong_answer += 1
                # Auto-add to failed file so future runs skip this problem
                fail_entry = {
                    "id": result["id"],
                    "image": result.get("image", ""),
                    "type": result.get("type", ""),
                    "answer": result.get("answer", ""),
                }
                fail_f.write(json.dumps(fail_entry, ensure_ascii=False) + "\n")
                fail_f.flush()
            elif not result.get("valid_format", False):
                invalid += 1
            else:
                correct += 1
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

    pbar.close()

    total = len(entries)
    print(f"\nDone! Processed {total} entries:")
    print(f"  Correct (saved):        {correct}")
    print(f"  Wrong answer (failed):  {wrong_answer}")
    print(f"  Invalid format:         {invalid}")
    print(f"  Errors:                 {errors}")
    print(f"Output: {output_path}")
    if wrong_answer:
        print(f"Added {wrong_answer} problems to {skip_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SFT reasoning responses using Gemini 2.5 Pro"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Input JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-pro",
        help="Gemini model name (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--max_output_tokens", type=int, default=8192,
        help="Max output tokens per response (default: 8192)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only first N entries, 0=all (default: 0)",
    )
    parser.add_argument(
        "--samples", type=int, default=1,
        help="Number of responses per problem (default: 1). "
             "When > 1, default temperature becomes 0.7.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, ignore existing output",
    )
    args = parser.parse_args()

    # Default temperature to 0.7 for multi-sample, unless explicitly set
    temperature = args.temperature
    if args.samples > 1 and temperature == 0.0:
        temperature = 0.7
        print(f"Multi-sample mode: using temperature={temperature}")

    asyncio.run(main(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        concurrency=args.concurrency,
        max_output_tokens=args.max_output_tokens,
        temperature=temperature,
        resume=not args.no_resume,
        limit=args.limit,
        samples=args.samples,
    ))

