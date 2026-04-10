"""
Skill Intelligence Architect: Data Preparation & Synthetic Distillation
Description: 
1. Generates skill rubrics using a Teacher model (Claude 3.5 Sonnet).
2. Orchestrates massive batch labeling for Job Descriptions.
3. Formats distilled reasoning into DeepSeek-R1 <think> format for training.
"""

import os
import re
import json
import time
import pandas as pd
import anthropic
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw_job_descriptions.parquet"
RUBRIC_PATH = DATA_DIR / "master_rubrics.json"
TRAIN_JSONL = DATA_DIR / "deepseek_train.jsonl"
EVAL_JSONL = DATA_DIR / "deepseek_eval.jsonl"

# Anthropic Configuration
MODEL_TEACHER = "claude-sonnet-4-6'" # Updated to standard public name
API_KEY = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=API_KEY)

# ==========================================
# 2. RUBRIC GENERATION (Teacher's Syllabus)
# ==========================================
def generate_skill_rubrics(skills_list):
    """Generates a universal 1-5 complexity rubric for target skills."""
    print(f"🧬 Generating Teacher Rubrics for {len(skills_list)} skills...")
    
    prompt = f"""
    Act as a Senior Global Skills Architect. Create a master universal difficulty rubric for these skills:
    {", ".join(skills_list)}

    RUBRIC RULES:
    - Define Levels 1-5 for EACH skill based on technical/conceptual complexity.
    - Level 3 in one skill must be as difficult as Level 3 in another.
    - Return ONLY valid JSON: {{"Skill Name": {{"1": "Def", "2": "Def", ...}}}}
    """

    response = client.messages.create(
        model=MODEL_TEACHER,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Robust JSON extraction
    text = response.content.text
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        rubrics = json.loads(json_match.group(1))
        with open(RUBRIC_PATH, 'w') as f:
            json.dump(rubrics, f, indent=4)
        return rubrics
    return None

# ==========================================
# 3. BATCH PROCESSING LOGIC
# ==========================================
def create_anthropic_batch(df, rubrics):
    """Prepares the 'Big Feed' for Anthropic's Message Batch API."""
    batch_requests = []
    
    system_prompt = (
        "You are a Senior Skills Evaluator. Assign a difficulty level (1-5) "
        "to a skill within a JD. Use the provided rubric and explain your reasoning."
    )

    for idx, row in df.iterrows():
        skill_rubric = rubrics.get(row['skill_name'], "Generic 1-5 scale")
        custom_id = f"{row['job_id']}__{re.sub(r'[^a-zA-Z0-9]', '_', row['skill_name'])}"[:64]

        user_content = f"""
        SKILL RUBRIC: {json.dumps(skill_rubric)}
        CONTEXT:
        - Title: {row['job_title']}
        - Skill: {row['skill_name']}
        - Responsibilities: {row['key_responsibilities']}
        """

        batch_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL_TEACHER,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}]
            }
        })
    return batch_requests

# ==========================================
# 4. SALVAGE & CLEANING (The 'Janitor' Logic)
# ==========================================
def salvage_llm_output(raw_output):
    """Safely extracts score and reasoning even if JSON is malformed."""
    # Attempt JSON extraction
    json_match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data.get("difficulty_level"), data.get("reasoning")
        except: pass

    # Heuristic Fallback
    level_match = re.search(r'level[:\s]*(\d)', raw_output, re.IGNORECASE)
    level = int(level_match.group(1)) if level_match else None
    reasoning = raw_output[:300] + "..." # Truncate for safety
    return level, reasoning

# ==========================================
# 5. R1 FORMATTING (Knowledge Distillation)
# ==========================================
def format_for_deepseek_r1(row):
    """Wraps teacher logic in R1-style <think> blocks."""
    instruction = f"Evaluate the difficulty (1-5) for skill: '{row['skill_name']}'."
    user_input = f"Job Responsibilities: {row['key_responsibilities']}"
    
    # The 'Reasoning-First' output structure
    output = (
        f"<think>\n{row['teacher_reasoning']}\n</think>\n"
        f"Final Difficulty Level: {int(row['difficulty_level'])}"
    )
    
    return {"instruction": instruction, "input": user_input, "output": output}

# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Load local data (Removed internal S3 paths)
    if not RAW_DATA_PATH.exists():
        print("❌ Please place your raw_job_descriptions.parquet in the /data folder.")
        exit()

    df = pd.read_parquet(RAW_DATA_PATH)
    
    # 2. Get Rubrics
    unique_skills = df['skill_name'].unique().tolist()
    rubrics = generate_skill_rubrics(unique_skills)

    # 3. Stratified Split & Final Formatting
    # (Assuming batch results are already merged as 'final_df' for this snippet)
    # ... logic for batch submission/retrieval ...

    print("📊 Preparing training records...")
    # Stratified Split (establishing Level 5 representation)
    train_df, eval_df = train_test_split(df, test_size=0.1, stratify=df['difficulty_level'])
    
    # Export to JSONL for Student Training
    train_records = train_df.apply(format_for_deepseek_r1, axis=1).tolist()
    with open(TRAIN_JSONL, 'w') as f:
        for entry in train_records:
            f.write(json.dumps(entry) + '\n')

    print(f"🚀 Success: Training data ready at {TRAIN_JSONL}")
