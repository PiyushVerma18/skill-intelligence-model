"""
Skill Intelligence Architect: R1-Reasoning Distillation Script
Author: Piyush Verma
Framework: Unsloth + PEFT/LoRA
Compute: Optimized for SageMaker/CUDA 12.8
"""

import os
import sys
import builtins
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from unsloth import FastLanguageModel

# ==========================================
# 1. 2026 RUNTIME COMPATIBILITY LAYER
# ==========================================
def apply_2026_patches():
    """Fixes legacy decorator and Pytree issues in the 2026 Python runtime."""
    builtins.strict = getattr(builtins, "strict", False)

    def robust_auto_docstring(*args, **kwargs):
        if len(args) == 1 and callable(args) and not kwargs: return args
        return lambda obj: obj
    
    builtins.auto_docstring = robust_auto_docstring
    
    try:
        import torch.utils._pytree as _pytree
        if not hasattr(_pytree, "register_constant"):
            _pytree.register_constant = lambda x: x
    except ImportError:
        pass
    print("✅ System: 2026 Compatibility Patches Applied.")

# ==========================================
# 2. CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
    "model_name": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "lora_rank": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "max_steps": 400,
    "batch_size": 2,
    "grad_accum_steps": 4,
    "output_dir": "skill_intel_outputs",
    "save_path": "skill_intelligence_r1_adapter_upd"
}

# ==========================================
# 3. TRAINING PIPELINE
# ==========================================
def run_training():
    apply_2026_patches()

    # Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = CONFIG["model_name"],
        max_seq_length = CONFIG["max_seq_length"],
        load_in_4bit = CONFIG["load_in_4bit"],
    )

    # Apply LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = CONFIG["lora_rank"],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = CONFIG["lora_alpha"],
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # Dataset Loading
    # Note: Ensure these paths are updated for your environment
    dataset = load_dataset("json", data_files={
        "train": "data/deepseek_train.jsonl",
        "test": "data/deepseek_eval.jsonl"
    })

    def formatting_prompts_func(examples):
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{tokenizer.eos_token}"
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        max_seq_length = CONFIG["max_seq_length"],
        args = TrainingArguments(
            per_device_train_batch_size = CONFIG["batch_size"],
            gradient_accumulation_steps = CONFIG["grad_accum_steps"],
            warmup_steps = 10,
            max_steps = CONFIG["max_steps"],
            learning_rate = CONFIG["learning_rate"],
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = CONFIG["output_dir"],
        ),
    )

    print("🚀 Training: Starting Knowledge Distillation...")
    trainer.train()

    # Save Results
    model.save_pretrained(CONFIG["save_path"])
    tokenizer.save_pretrained(CONFIG["save_path"])
    print(f"🏁 Success: Weights saved to {CONFIG['save_path']}")

# ==========================================
# 4. ANALYTICS: LOSS CURVE GENERATION
# ==========================================
def plot_loss(log_dir):
    """Generates training convergence plots from trainer_state.json."""
    log_file = os.path.join(log_dir, f"checkpoint-{CONFIG['max_steps']}/trainer_state.json")
    
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data["log_history"])
        train_loss = df[df['loss'].notna()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss['step'], train_loss['loss'], label='Training Loss', color='#2ecc71')
        plt.title('DeepSeek-R1 Training Convergence')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('docs/loss_curve.png') # Save for README
        print("📉 Analytics: Loss curve saved to docs/loss_curve.png")

if __name__ == "__main__":
    run_training()
    plot_loss(CONFIG["output_dir"])
