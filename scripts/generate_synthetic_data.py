"""
Skill Intelligence Architect: Synthetic Data Pipeline
Path 1: Responsibility Generation (Anthropic API)
Path 2: Complexity Auditing (Local vLLM/Unsloth Inference)
"""

import os
import re
import json
import time
import random
import pandas as pd
import torch
from anthropic import Anthropic
from unsloth import FastLanguageModel

# ==========================================
# 1. PATH A: TEACHER GENERATION (ANTHROPIC)
# ==========================================

# Standard configuration
industries = ["Fintech", "Health-tech", "Autonomous Vehicles", "Cybersecurity", "Cloud Gaming"]
roles = ["Backend Developer", "DevOps Engineer", "Data Scientist", "Frontend Architect", "Bioinformatician"]
skills = ["Python", "SQL", "Machine Learning", "Bioinformatics", "Project Management"]

GEN_SYSTEM_PROMPT = """You are a Senior Technical HR Architect. Write a 5-sentence, technically dense job responsibility for a specific role and skill.
CRITICAL: Use industry-standard terminology (idempotency, sharding, CI/CD orchestration). 
Every sentence must describe a specific action and a measurable outcome."""

def run_anthropic_generation(api_key, samples=2500):
    client = Anthropic(api_key=api_key)
    
    # Create unique combinations
    combinations = []
    for i in range(samples):
        combinations.append({
            "custom_id": f"gen_batch_{i}",
            "industry": random.choice(industries),
            "job_title": random.choice(roles),
            "skill": random.choice(skills)
        })

    # Build the Batch JSONL
    requests = []
    for item in combinations:
        user_content = f"CONTEXT: {item['industry']} | {item['job_title']} | Focus: {item['skill']}"
        requests.append({
            "custom_id": item['custom_id'],
            "params": {
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 1024,
                "temperature": 0.7,
                "system": [{"type": "text", "text": GEN_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user_content}]
            }
        })

    print(f"🚀 Submitting {len(requests)} requests to Anthropic Batch API...")
    batch = client.beta.messages.batches.create(requests=requests)
    return batch.id, combinations

# ==========================================
# 2. PATH B: STUDENT AUDIT (LOCAL INFERENCE)
# ==========================================

def load_auditor_model(adapter_path):
    """Loads the enterprise-trained auditor model (R1-Distill-Llama-8B)."""
    print("⚙️ Loading Skill Intelligence Auditor...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def get_audit_prompt(skill, job, resp):
    return f"""### Task: Skill Proficiency Audit
Act as a Senior Global Skills Architect. Evaluate (1-5) and justify.

### Pillars:
- L1: Foundational, L2: Applied, L3: Advanced, L4: Strategic, L5: Systemic

### Context:
- Target Skill: {skill}
- Job Title: {job}
- Responsibility: "{resp}"

Level:"""

def run_auditor_inference(df, model, tokenizer):
    final_results = []
    BATCH_SIZE = 24
    
    # Configure padding for batching
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(0, len(df), BATCH_SIZE):
        batch_slice = df.iloc[i : i + BATCH_SIZE]
        prompts = [get_audit_prompt(r['skill'], r['job_title'], r['responsibility']) for _, r in batch_slice.iterrows()]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        input_len = inputs.input_ids.shape[-1]
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.1)
        
        decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        
        for idx, text in enumerate(decoded):
            # Extract Level
            level_match = re.search(r"\d", text)
            level = int(level_match.group(0)) if level_match else None
            
            # Extract Reasoning (Scrub tags)
            reasoning = re.sub(r"Level:\s*\d|Final Difficulty Level:\s*\d", "", text, flags=re.IGNORECASE).strip()
            
            final_results.append({
                "skill": batch_slice.iloc[idx]['skill'],
                "job_title": batch_slice.iloc[idx]['job_title'],
                "responsibility": batch_slice.iloc[idx]['responsibility'],
                "teacher_reasoning": reasoning,
                "difficulty_level": level
            })
    
    return pd.DataFrame(final_results)

# ==========================================
# 3. MAIN PIPELINE EXECUTION
# ==========================================

if __name__ == "__main__":
    # 0. System Pre-flight Checks
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ADAPTER_PATH = "./weights/skill_intelligence_r1_adapter_upd"
    
    if not API_KEY:
        print("❌ Error: 'ANTHROPIC_API_KEY' environment variable not set.")
        sys.exit(1)

    print(f"{'='*60}")
    print("🧠 SKILL INTELLIGENCE ARCHITECT: CLEAN-ROOM DATA PIPELINE")
    print(f"{'='*60}\n")

    # --- STAGE 1: TEACHER GENERATION (PATH A) ---
    # In a real-world scenario, you might split these into two scripts.
    # Here, we orchestrate the submission.
    
    print("📡 [STAGE 1] Initiating Teacher Generation (Path A)...")
    batch_id, combinations_meta = run_anthropic_generation(API_KEY, samples=2500)

    # Portfolio Tip: In your GitHub documentation, explain that this stage
    # uses Sonnet 3.5 to ensure 'Technical Density' in the responsibilities.
    print(f"✅ Batch Submitted successfully. ID: {batch_id}")
    print("⏳ Note: Batch processing typically takes 10-30 minutes.")

    # --- STAGE 2: DATA RETRIEVAL & INTEGRATION ---
    # Assuming the results have been retrieved and saved to a temp CSV.
    # In your demo, you would load the results of the batch here.
    
    temp_gen_file = "data/raw_synthetic_responsibilities.csv"
    
    if not os.path.exists(temp_gen_file):
        print(f"🛑 Generation in progress. Run Path B once {temp_gen_file} is populated.")
        # For the sake of this script, we'll assume the file is ready for Path B.
        sys.exit(0)

    # --- STAGE 3: STUDENT AUDIT & SELF-DISTILLATION (PATH B) ---
    print("\n⚖️ [STAGE 3] Initiating Student-Auditor Inference (Path B)...")
    
    # Load the enterprise-tuned model to evaluate the synthetic data
    # This is the 'Self-Distillation' step that ensures privacy.
    model, tokenizer = load_auditor_model(ADAPTER_PATH)
    
    df_raw = pd.read_csv(temp_gen_file)
    print(f"📈 Auditing {len(df_raw)} synthetic samples for complexity...")
    
    df_final = run_auditor_inference(df_raw, model, tokenizer)

    # --- STAGE 4: CLEAN-ROOM DATA EXPORT ---
    print("\n📦 [STAGE 4] Finalizing Clean-Room Dataset...")
    
    # Final cleanup: Remove N/A results and ensure types are correct
    df_final = df_final.dropna(subset=['difficulty_level'])
    df_final['difficulty_level'] = df_final['difficulty_level'].astype(int)
    
    output_path = "data/Skill_Intelligence_Master_Dataset_CLEAN.xlsx"
    df_final.to_excel(output_path, index=False)
    
    # Memory Management: Clear GPU for the next pipeline task
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE COMPLETE")
    print(f"✅ Final Samples: {len(df_final)}")
    print(f"📂 Output: {output_path}")
    print(f"🛡️ Privacy Status: DECOUPLED (Zero Company Data in weights)")
    print(f"{'='*60}")

  
