
#!/usr/bin/env python3
"""
Simplified HuggingFace model wrapper for GPU inference.
Can be called like other model functions in model.py
"""

import json
import importlib.util
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables (for HF_TOKEN)
load_dotenv()

# Login to HuggingFace if token is available
_hf_token = os.getenv("HF_TOKEN")
if _hf_token and _hf_token != "your_huggingface_token_here":
    login(token=_hf_token)
    print("HuggingFace: Logged in with token.")
elif _hf_token == "your_huggingface_token_here":
    print("HuggingFace: No token configured (set HF_TOKEN in .env for gated models).")

# Global cache to avoid reloading model between calls
_model_cache = {}
_tokenizer_cache = {}

MODEL_NAME = os.getenv("HF_MODEL_NAME", os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B"))
MAX_NEW_TOKENS = 2048


def _load_model_once(model_name: str = MODEL_NAME):
    """Load model and tokenizer once, cache for reuse."""
    if model_name in _model_cache:
        return _model_cache[model_name], _tokenizer_cache[model_name]

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    has_accelerate = importlib.util.find_spec("accelerate") is not None
    if has_accelerate:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    if not has_accelerate:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    model.eval()

    print("Model loaded.")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            f"GPU memory allocated: "
            f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
        )

    _model_cache[model_name] = model
    _tokenizer_cache[model_name] = tokenizer

    return model, tokenizer


def call_model_hf(prompt: str, model_name: str = MODEL_NAME, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Call a HuggingFace model for inference.
    Simple callable interface like call_model_ollama in model.py.

    Args:
        prompt: The prompt text to send to the model
        model_name: HuggingFace model identifier (default: Qwen/Qwen3-1.7B)
        max_new_tokens: Maximum tokens to generate

    Returns:
        The model's response as a string
    """
    model, tokenizer = _load_model_once(model_name)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )

    return response.strip()


def call_model_hf_SoO(prompt: str, model_name: str = MODEL_NAME, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Call a HuggingFace model for SoO method (with specialized system prompt).
    Similar to call_model_ollama_SoO in model.py.

    Args:
        prompt: The prompt text to send to the model
        model_name: HuggingFace model identifier (default: Qwen/Qwen3-1.7B)
        max_new_tokens: Maximum tokens to generate

    Returns:
        The model's response as a string
    """
    model, tokenizer = _load_model_once(model_name)

    messages = [
        {
            "role": "system",
            "content": "You are an expert at understanding human communication. Please leverage the Information provided and choose the most probable answer to the question from the options. Output your final answer by strictly following this format: [A], [B], [C], or [D]"
        },
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )

    return response.strip()


# Keep the original main() for standalone usage
if __name__ == "__main__":
    INPUT_FILE = "prompts.jsonl"
    OUTPUT_FILE = "results.jsonl"

    print("=" * 60)
    print("Environment information")
    print("=" * 60)

    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    model, tokenizer = _load_model_once(MODEL_NAME)

    prompts = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} prompts")

    output_path = Path(OUTPUT_FILE)
    completed_ids = set()

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

        print(
            f"Found existing results. "
            f"Skipping {len(completed_ids)} completed samples."
        )

    with open(output_path, "a", encoding="utf-8") as out_file:

        for idx, sample in enumerate(prompts):

            sample_id = sample["id"]

            if sample_id in completed_ids:
                continue

            prompt = sample["prompt"]

            start = time.time()

            try:

                response = call_model_hf(
                    prompt=prompt,
                    model_name=MODEL_NAME,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                elapsed = time.time() - start

                result = {
                    "id": sample_id,
                    "prompt": prompt,
                    "response": response,
                    "latency_seconds": elapsed,
                    "model": MODEL_NAME,
                }

                out_file.write(
                    json.dumps(result, ensure_ascii=False)
                    + "\n"
                )

                out_file.flush()

                print(
                    f"[{idx+1}/{len(prompts)}] "
                    f"id={sample_id} "
                    f"time={elapsed:.2f}s"
                )

            except Exception as e:

                error_result = {
                    "id": sample_id,
                    "prompt": prompt,
                    "error": str(e),
                    "model": MODEL_NAME,
                }

                out_file.write(
                    json.dumps(error_result, ensure_ascii=False)
                    + "\n"
                )

                out_file.flush()

                print(
                    f"[ERROR] sample {sample_id}: {e}"
                )

    print("Benchmark completed.")

