import os
import re
import json
import torch
import gc
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

class SkillIntelligenceInference:
    def __init__(self, model_id, lora_path, cache_dir="/tmp/huggingface"):
        self.model_id = model_id
        self.lora_path = lora_path
        self._apply_environment_patches()
        
        self.llm = LLM(
            model=self.model_id,
            download_dir=cache_dir,
            enable_lora=True,
            max_lora_rank=16,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            enforce_eager=True
        )

    def _apply_environment_patches(self):
        """Fixes GLIBCXX and C++ library conflicts in SageMaker/Conda environments."""
        conda_prefix = os.environ.get('CONDA_PREFIX', '/opt/conda')
        os.environ["LD_PRELOAD"] = f"{conda_prefix}/lib/libstdc++.so.6"
        print("✅ Environment: GLIBCXX bridge active.")

    def parse_r1_response(self, raw_text):
        """Extracts DeepSeek-R1 thought process and numeric score."""
        # 1. Extract Reasoning
        if "</think>" in raw_text:
            parts = raw_text.split("</think>")
            thought = parts.replace("<think>", "").strip()
        else:
            thought = "No reasoning block found."

        # 2. Extract Difficulty Level
        score_match = re.search(r"Final Difficulty Level:\s*(\d)", raw_text)
        level = int(score_match.group(1)) if score_match else "Unknown"
        
        return level, thought

    def predict_batch(self, prompts, temperature=0.6):
        """Executes high-throughput batch inference."""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=512,
            stop=["### Instruction:"]
        )
        
        print(f"⚡ Processing batch of {len(prompts)} requests...")
        lora_req = LoRARequest("skill_intel_lora", 1, self.lora_path)
        outputs = self.llm.generate(prompts, sampling_params, lora_request=lora_req)
        
        results = []
        for output in outputs:
            raw_text = output.outputs.text
            level, reasoning = self.parse_r1_response(raw_text)
            results.append({
                "level": level,
                "reasoning": reasoning,
                "raw": raw_text
            })
        return results

if __name__ == "__main__":
    # Local Testing Logic
    ENGINE = SkillIntelligenceInference(
        model_id="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
        lora_path="./weights/skill_intelligence_r1_adapter_upd"
    )
    
    test_resp = "Own the end-to-end lifecycle of client COA libraries..."
    result = ENGINE.predict_batch([f"### Instruction:\nEvaluate skill: Product Management\n\n### Input:\n{test_resp}\n\n### Response:\n"])
    print(f"Result: {result['level']}")
