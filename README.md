# Reinforcement Learning via Evolutionary Fine-Tuning Loops (REFTL)
Official implementation of my thesis "Reinforcement Learning via Evolutionary Fine-Tuning Loops (REFTL): Reducing Large Language Model Bias through Self-Directed Alignment and Multi-Adapter Merging"

**Author:** Kandavel Yogeshram
**Supervisor:** Sakurai Kouichi

---

## Overview

Large Language Models (LLMs) frequently inherit social biases present in their training data. Traditional alignment methodologies like RLHF and DPO rely heavily on human-annotated preference datasets, which are expensive, bounded in scale, and GPU-memory intensive.

**REFTL** introduces a fully automated, self-directed framework designed to mitigate LLM social bias while preserving core reasoning capabilities. Instead of static fine-tuning, REFTL treats alignment as an evolutionary search over merged QLoRA adapters.

---

## Key Features

* **Self-Directed Data Generation:** Automated synthesis of question-answer pairs using MinHash Locality-Sensitive Hashing (LSH) to prevent semantic redundancy.
* **Quality Filtering & Self-Evaluation:** Prompt-driven debiased persona generation evaluated by a model judge and filtered using a Number Density Penalty Function to prevent structural output degeneration.
* **Parallel Evolutionary Loops:** Optimization structured across 10 outer iterations and 3 parallel evolutionary nodes to explore diverse alignment trajectories.
* **DARE-TIES Multi-Adapter Merging:** Combines the historical trajectory of champion QLoRA adapters using Drop and Rescale (DARE) and TIES sign consensus to eliminate parameter interference.
* **Resource Efficient:** Uses 4-bit NormalFloat (nf4) QLoRA parameter-efficient fine-tuning capable of running on single consumer GPU hardware.

---

## Benchmark Results

Evaluated on **Meta Llama-3.2-1B-Instruct** across the **BBQ** (Bias Benchmark for QA) and **MMLU** (Massive Multitask Language Understanding) benchmarks:

| Model Configuration | BBQ Total Acc. (%) | Ambiguous Context Acc. (%) | Ambiguous Bias Score | MMLU Avg. Acc. (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Base Model (Llama-3.2-1B)** | 29.01% | 10.85% | 0.04247 | 36.61% |
| **REFTL (Merge Weight 1.0)** | 30.26% | 15.32% | 0.03379 | 36.80% |
| **REFTL (Merge Weight 2.0)** | 32.97% | 24.11% | 0.01764 | 36.55% |

* **Bias Reduction:** REFTL with merge weight 2.0 reduces the BBQ ambiguous bias score from 0.04247 down to 0.01764 while more than doubling ambiguous context accuracy (10.85% to 24.11%).
* **Capability Preservation:** MMLU accuracy remains intact (36.55% vs. 36.61% baseline), confirming that DARE-TIES adapter merging prevents catastrophic forgetting in general academic domains.

---

## Hardware Requirements

The experimental setup was conducted on local desktop hardware:
* **CPU:** Intel Core i7-10700 @ 2.90GHz
* **RAM:** 16.0 GB
* **GPU:** NVIDIA GeForce RTX 2060 SUPER
* **Total Runtime:** ~18–19 hours with automated thermal management

## Notes
While Training and Evaluation
In BBQ_Data folder there was files from https://github.com/nyu-mll/BBQ/tree/main/data and https://github.com/nyu-mll/BBQ/blob/main/supplemental/additional_metadata.csv
In MMLU_Data folder there was data files from https://github.com/hendrycks/test 's test download link.
