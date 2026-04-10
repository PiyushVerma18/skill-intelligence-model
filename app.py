import os
import torch
import re
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. SECURITY & CONFIGURATION
# Access the Secret token you added in the Space Settings
HF_TOKEN = os.getenv("HF_TOKEN")

# The public base model
BASE_MODEL_ID = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"

# YOUR PRIVATE MODEL ID (Change 'your-username' to your HF username)
LORA_ID = "piyushverma23/skill-intelligence-r1-adapter"

# 2. OPTIMIZED 4-BIT LOAD (NVIDIA T4 Compatible)
print("⚙️ Initializing Skill Intelligence Engine...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load Tokenizer & Base Model with the Secret Token
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)

# Load your Private Adapter
print("🧠 Applying private adapter weights...")
model = PeftModel.from_pretrained(base_model, LORA_ID, token=HF_TOKEN)

# 3. TEXT CLEANING HELPER
def clean_text(text):
    """Removes byte-level artifacts and fixes encoding issues."""
    replacements = {
        'âĢĶ': '—', 'âĢĵ': '–', 'âĢĻ': "'", 
        'âĢś': '"', 'âĢť': '"', 'Ċ': '\n', 'Ġ': ' '
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove 'Final Difficulty' header if it bleeds into the reasoning
    text = re.sub(r"^\s*(Final\s+)?Difficulty\s*[:\-]*\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()

# 4. THE CORE ANALYSIS FUNCTION
def evaluate_skill(job_title, skill, responsibility):
    if not skill or not responsibility:
        return "Error", "Missing input fields."

    prompt = f"### Instruction:\nYou are a technical skills architect. Evaluate the difficulty level (1-5) for the skill: '{skill}'.\n\n### Input:\nJob Title: {job_title}\nKey Responsibilities: {responsibility}\n\n### Response:\n<think>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = inputs.input_ids.shape[-1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2048, 
            temperature=0.6, 
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    # Slice output to get only the response
    generated_tokens = outputs[0, prompt_length:]
    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

    # Parsing Reasoning vs Conclusion
    if "</think>" in full_response:
        parts = full_response.split("</think>")
        thought = parts[0].replace("<think>", "").strip()
        conclusion = parts[0].strip() if len(parts[0]) > 1 else ""
    else:
        thought = "No reasoning tag found."
        conclusion = full_response[0].strip()

    thought = clean_text(thought)
    conclusion = clean_text(conclusion)
    full_response = clean_text(full_response)

    # Extract Score using Regex
    score_patterns = [
        r"Final Difficulty(?:\s*Level)?:\s*(\d)", 
        r"Level:\s*(\d)",
        r"Difficulty:\s*(\d)"
    ]

    predicted_level = "Unknown"
    # Search in the conclusion (the part after the thought)
    for pattern in score_patterns:
        match = re.search(pattern, conclusion, re.IGNORECASE)
        if match:
            predicted_level = match.group(1)
            break

    # Fallback: Search the thought process if the conclusion is empty/missing score
    if predicted_level == "Unknown":
        for pattern in score_patterns:
            match = re.search(pattern, full_response, re.IGNORECASE)
            if match:
                predicted_level = match.group(1)
                break
  
    return f"Level {predicted_level}", thought

# 5. GRADIO UI SETUP
demo = gr.Interface(
    fn=evaluate_skill,
    inputs=[
        gr.Textbox(label="Job Title", placeholder="e.g., Senior ML Engineer"),
        gr.Textbox(label="Skill to Evaluate", placeholder="e.g., Python"),
        gr.Textbox(label="Job Responsibilities", lines=8, placeholder="Paste JD snippets here...")
    ],
    outputs=[
        gr.Label(label="Complexity Score"),
        gr.Textbox(label="R1 Reasoning Process", lines=12)
    ],
    title="🧠 Skill Intelligence Architect",
    description="Contextual skill difficulty estimation (1-5) using fine-tuned R1-Distill-Llama-8B.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # HF Spaces handles sharing automatically, so share=True is not needed
    demo.launch()

