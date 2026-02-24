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
| **Accuracy** | 47.0% | 53% | 57% |
| **Parse success rate** | 98.33% | 99.33%| 99.33% |
| **Average completion length** | 369 | 346 | 317 |
| **reward_by_source/geoqa** | 0.429 ± 0.436  | 0.468 ± 0.445| 0.491 ± 0.447 |
| **reward_by_source/synthesis** | 0.631 ± 0.444 |0.711 ± 0.420 | 0.756 ± 0.401|
| **accuracy/geoqa/difficulty 1** | 0.5098 | 0.5490 | 0.5490 |
| **accuracy/geoqa/difficulty 2** | 0.3594 | 0.4219 | 0.4375 |
| **accuracy/geoqa/difficulty 3** | 0.2292 | 0.2500 | 0.3125 |
| **accuracy/synthesis/difficulty 1** | 0.6512 | 0.7442 | 0.7209 |
| **accuracy/synthesis/difficulty 2** | 0.5893 | 0.6250 | 0.7321 |
| **accuracy/synthesis/difficulty 3** | 0.5263| 0.6842 | 0.7368 |
| **accuracy/geoqa/difficulty 1** | 0.559 ± 0.450 | 0.594 ± 0.448 | 0.594 ± 0.448 |
| **accuracy/geoqa/difficulty 2** | 0.419 ± 0.436  | 0.477 ± 0.447 | 0.492 ± 0.448|
| **accuracy/geoqa/difficulty 3** | 0.304 ± 0.380  | 0.323 ± 0.391 | 0.381 ± 0.417 |
| **reward/synthesis/difficulty 1** | 0.686 ± 0.429  | 0.770 ± 0.393| 0.749 ± 0.404 |
| **reward/synthesis/difficulty 2** |0.627 ± 0.447 | 0.663 ± 0.436 | 0.757 ± 0.402 |
| **reward/synthesis/difficulty 3** | 0.574 ± 0.449  | 0.716 ± 0.418 | 0.763 ± 0.396 |
