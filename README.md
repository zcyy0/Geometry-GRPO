# 📐 Visual Geometry Reasoning with Qwen2.5-VL & GRPO

![Status](https://img.shields.io/badge/Status-Training_In_Progress-yellow)
![Model](https://img.shields.io/badge/Base_Model-Qwen_2.5_VL_3B-green)
![Tech](https://img.shields.io/badge/Stack-TRL_%7C_VLLM_%7C_LoRA-blue)

## 📌 Project Overview
This project implements **Group Relative Policy Optimization (GRPO)** to enhance **visual geometry reasoning** in the **Qwen2.5-VL-3B-Instruct** model. 

Unlike standard fine-tuning, this pipeline uses Reinforcement Learning (RL) to enforce verifiable "Chain of Thought" (CoT) reasoning. The training system leverages **HuggingFace TRL** for the RL loop, **LoRA** for parameter-efficient tuning, and **VLLM** for high-throughput generation during the exploration phase.

**Target Benchmark:** [MathVision](https://huggingface.co/datasets/mathvision/mathvision)  
**Training Data:** [VLAA-Thinking (GeoQA/Synthesis)](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) & [Zebra CoT Geometry](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)

---

## 🔬 Methodology: Structure-Aware Reward Modeling
The goal is to bootstrap the model's reasoning capabilities by strictly enforcing a structured output format:
`<think>...reasoning steps...</think><answer>...final answer...</answer>`

### The Reward Function
To penalize "shortcut learning" (guessing without reasoning) while maintaining training stability, I implemented a hierarchical reward function.

R(y) = 1.0 if if Correct Answer AND Strict Format; 0.1 if Incorrect Answer BUT Strict Format; 0.0 otherwise

---

## 🛠️ Data Engineering & Curriculum
This project moves beyond simple dataset loading by implementing a **Curriculum Learning** strategy based on problem complexity.

### Data Processing Pipeline
* **Normalization:** Converted VLAA-GeoQA's multiple-choice format `(A/B/C/D)` into open-form expressions.
* **Math Standardization:** Normalized all ground truth values to standard LaTeX math expressions using `math_verify` equivalence checks.
* **Difficulty Stratification:** Classified VLAA and Zebra-COT examples into 3 tiers based on reasoning length and ground-truth complexity:
    * **Tier 1 (Easy):** Bottom 30% difficulty
    * **Tier 2 (Medium):** 30%-70% difficulty
    * **Tier 3 (Hard):** Top 30% difficulty

### Project Structure
```bash
├── train/
│   └── train_grpo.py           # Main RL training loop (TRL + VLLM integration)
├── scripts/
│   ├── build_splits.py         # Stratified splitting (Train: 5k, Dev: 300, Test: 1k)
│   ├── process_geoqa_data.py   # Multiple-choice to Open-ended conversion
│   └── process_zebra_cot_geometry_data.py    # Geometry dataset cleaning
│   └── process_synthesis_data.py
├── utils/
│   └── extract_answer.py       # Regex logic for answer extraction & normalization
└── README.md
```
## Results
The curriculum learning is divided into three phases: phase 1 trains on all difficulty 1 questions; phase 2 difficulty 2 questions and phase 3 difficulty 3 questions. 
Weights and Bias charts on rewards of the three phases:
| Phase 1 | Phase 2 | Phase 3 |
| :---: | :---: | :---: |
| ![Run 1](./assets/phase1_accuracy_reward_mean.png) | ![Run 2](./assets/phase2_accuracy_reward_mean.png) | ![Run 3](./assets/phase3_accuracy_reward_mean.png) |

At the end of each phase, the model is evaluated on 300 validation examples. The results are as following:
| Metric | Phase 1| Phase 2| Phase 3|
| :--- | :---: | :---: | :---: |
| **Accuracy** | 89.4% | **91.2%** | 90.1% |
| **Parse success rate** | 0.245 | 0.210 | **0.198** |
| **Average completion length** | 50 | 42 | 45 |
| **reward_by_source/geoqa** | 1e-4 | 5e-5 | 1e-4 |
| **accuracy/geoqa/difficulty 1** | 32 | 32 | 64 |
| **accuracy/geoqa/difficulty 2** | 32 | 32 | 64 |
| **accuracy/geoqa/difficulty 3** | 32 | 32 | 64 |
| **accuracy/synthesis/difficulty 1** | 32 | 32 | 64 |
| **accuracy/synthesis/difficulty 2** | 32 | 32 | 64 |
| **accuracy/synthesis/difficulty 3** | 32 | 32 | 64 |
| **accuracy/geoqa/difficulty 1** | 32 | 32 | 64 |
| **accuracy/geoqa/difficulty 2** | 32 | 32 | 64 |
| **accuracy/geoqa/difficulty 3** | 32 | 32 | 64 |
| **reward/synthesis/difficulty 1** | 32 | 32 | 64 |
| **reward/synthesis/difficulty 2** | 32 | 32 | 64 |
| **reward/synthesis/difficulty 3** | 32 | 32 | 64 |
