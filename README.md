# Geometry-GRPO
This repo studies how **verifiable RL (GRPO)** can improve **visual geometry reasoning** in **Qwen2.5-VL-3B-Instruct**. This project use the following setup:

- **Benchmark:** MathVision
- **Model:** Qwen2.5-VL-3B-Instruct 
- **Training dataset:**
  - VLAA-Thinking GeoQA170k and Synthesis subset
  - Zebra COT Geometry subset
- **Metric:** match accuracy + pass@k rate

## Roadmap 
- **[X]Stage 1 — Process data and build verifiable scoring**
  - Build data processing pipelines that convert VLAA-GeoQA multiple choice ground truth "(A/B/C/D)" to open-form expressions, and normalize the datasets' ground truths to standard latext math expressions.
  - Implement `math_verify`-based equivalence reward.
  - Classify dataset into 3 different difficulty levels (difficulty=1,2,3) based on the reference reasoning length and ground truth complexity.
  - Set up train/dev/test splits: training set: 5000 examples; dev set: 300 examples; test set: 1000 examples. Difficulty ratio: 20% difficulty= 3; 60% difficulty=2; 20% difficulty=1.

- **[In Progress ]Stage 2 — GRPO training**
  - Introduce curriculum learning: divide the training to 3 phases. Phase 1 trains on difficulty=1 data; phase 2 trains on difficulty=2 data; phase 3 trains on difficulty=3 data.
  - Prompt requires the model output to follow a strict format: <think>...</think><answer>...</answer>
  - Reward design: Outcome based reward. reward = 1 if the output follows strict format and the answer is correct; reward = 0.5 if the output only has answer tags and the answer is correct; reward = 0 for other cases.
  - Track: mean reward per source per difficulty level, accuracy, pass@k rate

- **[ ]Stage 3 — Evaluation on Benchmark**
- **[ ]Stage 4 — Denser reward**
  - Explore partial-credit reward
