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
