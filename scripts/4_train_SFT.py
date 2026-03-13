#!/usr/bin/env python3
"""SFT training for Qwen2.5-VL-3B on PGPS9K geometry using simplified Gemini responses.

Trains the model to produce step-by-step reasoning with a final answer
by imitating validated Gemini responses (simplified format: no <facts>/<theorems> tags).

Supports two modes:
  - Single-task (default): image + question → reasoning + answer
  - Multi-task (--multitask): 3 tasks (visual grounding, reasoning, end-to-end)

Launch (single GPU):
    python scripts/PGPS9K/4_train_sft.py
    python scripts/PGPS9K/4_train_sft.py --multitask

Launch (multi-GPU with accelerate):
    accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 \
        scripts/PGPS9K/4_train_sft.py --multitask

Resume from checkpoint:
    python scripts/PGPS9K/4_train_sft.py --resume_from outputs/PGPS9K/sft/checkpoint-500
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
torch.set_float32_matmul_precision("high")

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig

# Add project root to path for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.compare_answer import extract_answer, parse_numeric, parse_all_numeric, check_answer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(_PROJECT_ROOT)
DEFAULT_TRAIN = BASE / "data/PGPS9K/sft_responses_train_1500_simplified.jsonl"
DEFAULT_MULTITASK_TRAIN = BASE / "data/PGPS9K/sft_multitask_train.jsonl"
DEFAULT_MULTITASK_VAL = BASE / "data/PGPS9K/sft_multitask_val.jsonl"
DEFAULT_VAL = BASE / "data/PGPS9K/sft_responses_val_simplified.jsonl"
DEFAULT_OUTPUT = BASE / "outputs/PGPS9K/sft"

# ---------------------------------------------------------------------------
# System prompt (simplified format — no <facts>/<theorems>/<reasoning> tags)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You will be shown a geometry diagram image and a question. Solve it step by step.

Wrap your reasoning inside <think>...</think> tags. In your reasoning, describe the relevant geometric facts you observe in the diagram and the theorems you apply. Then give your final answer inside <answer>...</answer> tags using LaTeX notation, without units."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sft_data(
    path: Path,
    verify_answers: bool = True,
    max_examples: int = 0,
) -> list[dict]:
    """Load validated SFT responses, optionally verifying answer correctness."""
    records = []
    skipped_format = 0
    skipped_answer = 0
    skipped_image = 0
    skipped_error = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Skip errors
            if "_error" in obj:
                skipped_error += 1
                continue

            # Skip invalid format (check for <think> and <answer> tags)
            resp = obj.get("response", "")
            if "<think>" not in resp or "<answer>" not in resp:
                skipped_format += 1
                continue

            # Skip missing images
            image_path = obj.get("image", "")
            if not image_path or not os.path.exists(image_path):
                skipped_image += 1
                continue

            # Verify answer correctness
            if verify_answers:
                gold_raw = str(obj.get("answer", ""))
                pred_raw = extract_answer(obj.get("response", "")) or ""
                gold_val = parse_numeric(gold_raw)
                pred_vals = parse_all_numeric(pred_raw)
                if gold_val is None or not pred_vals or not check_answer(pred_vals, gold_val):
                    skipped_answer += 1
                    continue

            records.append(obj)
            if max_examples and len(records) >= max_examples:
                break

    print(f"Loaded {len(records)} valid examples from {path}")
    if skipped_format:
        print(f"  Skipped {skipped_format} (invalid format)")
    if skipped_answer:
        print(f"  Skipped {skipped_answer} (wrong answer)")
    if skipped_image:
        print(f"  Skipped {skipped_image} (missing image)")
    if skipped_error:
        print(f"  Skipped {skipped_error} (API errors)")

    return records


def strip_facts_from_prompt(prompt: str) -> str:
    """Strip the preamble, Topic, and Facts section from the prompt, keeping Question only."""
    for line in prompt.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("Question:"):
            return stripped
    return prompt.strip()


class ListDataset(torch.utils.data.Dataset):
    """Simple list-backed dataset that avoids PyArrow serialization issues
    with mixed content types in chat messages."""

    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def records_to_dataset(records: list[dict], processor) -> ListDataset:
    """Convert SFT records to a dataset with chat messages."""
    examples = []

    for rec in records:
        # Strip facts from prompt — model must extract facts from the image
        user_text = strip_facts_from_prompt(rec["prompt"])

        # Build chat messages: system + user (image + text) + assistant (response)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rec["image"]},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": rec["response"],
            },
        ]

        examples.append({
            "messages": messages,
            "images": [rec["image"]],
        })

    return ListDataset(examples)


def load_multitask_data(
    path: Path,
    max_examples: int = 0,
) -> ListDataset:
    """Load pre-built multi-task JSONL (messages already constructed).

    Each row has: messages, images, task, id.
    Task 2 (reasoning) has images=[] — no image input.
    """
    examples = []
    task_counts = {}
    skipped_image = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = obj.get("task", "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1

            # Verify images exist on disk
            skip = False
            for img_path in obj.get("images", []):
                if not os.path.exists(img_path):
                    skipped_image += 1
                    skip = True
                    break
            if skip:
                continue

            examples.append({
                "messages": obj["messages"],
                "images": obj["images"],
            })

            if max_examples and len(examples) >= max_examples:
                break

    print(f"Loaded {len(examples)} multi-task examples from {path.name}")
    for t, c in sorted(task_counts.items()):
        print(f"  {t}: {c}")
    if skipped_image:
        print(f"  Skipped {skipped_image} (missing image)")

    return ListDataset(examples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SFT training on PGPS9K with Gemini responses")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train_data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val_data", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to LoRA checkpoint to resume from")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Limit training examples (0 = all)")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_verify_answers", action="store_true",
                        help="Skip answer correctness verification during data loading")
    parser.add_argument("--multitask", action="store_true",
                        help="Use multi-task dataset (visual grounding + reasoning + end-to-end)")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="pgps9k-sft")

    args = parser.parse_args()

    # Wandb
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            group=os.environ.get("WANDB_RUN_GROUP"),
            name=os.environ.get("WANDB_RUN_NAME"),
        )
    except ImportError:
        print("wandb not available, skipping")

    # Processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_id, use_fast=True, padding_side="right",
    )

    # Load data
    if args.multitask:
        # Multi-task: load pre-built messages directly
        mt_path = args.train_data if "multitask" in str(args.train_data) else DEFAULT_MULTITASK_TRAIN
        print(f"Loading multi-task training data from {mt_path}...")
        train_ds = load_multitask_data(mt_path, max_examples=args.max_examples)
        # Multi-task val set
        mt_val_path = args.val_data if "multitask" in str(args.val_data) else DEFAULT_MULTITASK_VAL
        if mt_val_path.exists():
            print(f"Loading multi-task validation data from {mt_val_path}...")
            val_ds = load_multitask_data(mt_val_path)
        else:
            val_ds = None
    else:
        # Single-task: load simplified responses and build messages
        print(f"Loading training data from {args.train_data}...")
        train_records = load_sft_data(
            args.train_data,
            verify_answers=not args.no_verify_answers,
            max_examples=args.max_examples,
        )

        val_records = []
        if args.val_data.exists():
            print(f"Loading validation data from {args.val_data}...")
            val_records = load_sft_data(
                args.val_data,
                verify_answers=not args.no_verify_answers,
            )

        train_ds = records_to_dataset(train_records, processor)
        val_ds = records_to_dataset(val_records, processor) if val_records else None

    print(f"Train: {len(train_ds)} examples")
    if val_ds:
        print(f"Val: {len(val_ds)} examples")

    # Check how many examples exceed max_length after tokenization
    print(f"\nChecking sequence lengths (max_length={args.max_length})...")
    from qwen_vl_utils import process_vision_info
    seq_lengths = []
    truncated = 0
    for i in range(len(train_ds)):
        ex = train_ds[i]
        text = processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(ex["messages"])
        if image_inputs:
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
        else:
            inputs = processor(text=[text], return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]
        seq_lengths.append(seq_len)
        if seq_len > args.max_length:
            truncated += 1
    seq_lengths.sort()
    n = len(seq_lengths)
    print(f"  Min: {seq_lengths[0]}, Median: {seq_lengths[n//2]}, "
          f"P95: {seq_lengths[int(n*0.95)]}, Max: {seq_lengths[-1]}")
    print(f"  Exceed max_length ({args.max_length}): {truncated}/{n} "
          f"({truncated/n:.1%}) — these will lose EOS token")

    # Model
    print(f"Loading model: {args.model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16,
    )

    if args.resume_from:
        print(f"Loading LoRA checkpoint from {args.resume_from}...")
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        model.print_trainable_parameters()
    else:
        print("Applying fresh LoRA...")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Output directory
    output_dir = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Training config
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if val_ds else "no",
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=args.max_length,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # Trainer — pass full processor so TRL detects VLM and uses its built-in
    # DataCollatorForVisionLanguageModeling
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
    )

    # Train
    print(f"\nStarting SFT training:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} grad accum")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  Max seq length: {args.max_length}")
    print(f"  Output: {output_dir}")

    trainer.train()

    # Save final
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nTraining complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
