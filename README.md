# Skill Intelligence Architect (SIA)

Context-Aware Complexity Estimation via R1-Reasoning Distillation

📌 **Executive Summary**

The Skill Intelligence Architect (SIA) is an LLM-based system designed to solve the "Keyword Matching Fallacy" in talent intelligence. Traditional systems treat a skill (e.g., Python) as a binary attribute; SIA quantifies the complexity of that skill (Scale 1-5) by analyzing the surrounding context of a Job Description (JD).

By Distilling reasoning from high-parameter "Teacher" models into a specialized "Student" model (DeepSeek-R1-Distill-Llama-8B), SIA provides high-fidelity, explainable difficulty scores with low latency and inference cost.

🏗️ **Technical Architecture & Stack**

- Base Model: Llama-3-8B (vLLM Optimized)

- Fine-tuning Method: Parameter-Efficient Fine-Tuning (PEFT) using LoRA.

- Quantization: 4-bit NormalFloat (NF4) via bitsandbytes for efficient deployment.

- Orchestration: Fine-tuned on Amazon SageMaker; Deployed via Gradio on Hugging Face Spaces.

- Distillation Logic: Teacher-Student chain-of-thought (CoT) distillation.

🧪 **Methodology & Workflow**

**Phase 1: Knowledge Distillation (The Teacher)**

To bypass the lack of granularly labeled JD datasets, we utilized Anthropic (Claude 3.5 Sonnet) to generate a high-quality "Ground Truth" synthetic dataset.

- Rubric-Augmented Prompting: The Teacher model was provided with a strict 5-level rubric.

- Reasoning Extraction: Instead of simple labels, we captured the Chain-of-Thought (CoT), explaining why a responsibility aligns with a specific rubric level.

**Phase 2: Supervised Fine-Tuning (The Student)**

We fine-tuned the DeepSeek-R1-Distill-Llama-8B on 2,500+ high-fidelity samples.

- **Objective:** The Student was trained to predict the Reasoning Block + Difficulty Score.

- **Outcome:** By learning the logic behind the score, the model generalizes significantly better to unseen industries and emerging tech roles compared to standard classifiers.

  📑 Case Study: Contextual Intelligence in Action

  The model's strength lies in its ability to differentiate the same skill across different seniority tiers.

  Skill	Job Title	Responsibility Snippet	Score
Python	Junior Data Analyst	"Automating daily Excel reports and basic CSV cleaning."	Level 2
Python	ML Infrastructure Lead	"Designing custom CUDA kernels and optimizing distributed training for 70B+ models."	Level 5

