import os
from openai import OpenAI
# from huggingface_hub import InferenceClient
from dotenv import load_dotenv
# from ollama import chat
load_dotenv()

def call_model_ollama(prompt: str) -> str:
    """
    Call an Ollama model to get the answer.
    model: e.g. "qwen:4b", "qwen3:8b", etc.
    """

    response = chat(
        model='qwen:4b',  # change model if needed
        messages=[
            #{"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
    )

    return response.message.content

def call_model_ollama_SoO(prompt: str) -> str:
    """
    Call an Ollama model to get the answer.
    model: e.g. "qwen:4b", "qwen3:8b", etc.
    """

    response = chat(
        model='qwen:4b',  # change model if needed
        messages=[
            {"role": "system", "content": "You are an expert at understanding human communication. Please leverage the Information provided and choose the most probable answer to the question from the options. Output your final answer by strictly following this format: [A], [B], [C], or [D]"},
            {"role": "user", "content": prompt},
        ],
    )

    return response.message.content


def call_model_deepseek(prompt: str) -> str:
    """
    call deepseek API to get the answer.
    model: "deepseek-chat" (DeepSeek-V3.2) or "deepseek-reasoner" (DeepSeek-V3.2 thinking mode)
    """

    # for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
    client = OpenAI(api_key=os.getenv("deepseek_api"), base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
    ],
        max_tokens=1024,
        temperature=0.0,
        stream=False
    )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def call_model_qwen8b(prompt: str) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("openrouter_api"),
    )

    completion = client.chat.completions.create(
        # extra_headers={
        #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        #     "X-OpenRouter-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        # },
        extra_body={},
        model="qwen/qwen3-8b",
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content


"""
Model calling functions adapted for Google Colab with T4 GPU
Uses HuggingFace transformers for local model inference
Also supports GGUF models from HuggingFace
"""

import torch
import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import os
from huggingface_hub import hf_hub_download, list_repo_files

# Global cache for models to avoid reloading
_model_cache = {}
_tokenizer_cache = {}

def _package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def _manual_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer(model_name: str, use_4bit: bool = True, gguf_filename: Optional[str] = None):
    """
    Load or retrieve cached model and tokenizer.
    Uses 4-bit quantization to fit larger models on T4 GPU.
    Supports both standard HuggingFace models and GGUF models.
    """
    if model_name in _model_cache:
        return _model_cache[model_name], _tokenizer_cache[model_name]

    print(f"Loading model: {model_name}...")

    accelerate_available = _package_available("accelerate")
    can_use_device_map = torch.cuda.is_available() and accelerate_available
    device_map = "auto" if can_use_device_map else None

    if torch.cuda.is_available() and not accelerate_available:
        print("  accelerate not found; loading without device_map/4-bit and moving model to CUDA manually.")

    # Check if this is a GGUF model
    is_gguf = "GGUF" in model_name.upper() or (gguf_filename and gguf_filename.endswith(".gguf"))

    if is_gguf:
        # For GGUF models, we need to find the actual GGUF file
        # First, try to list files in the repo and find a suitable GGUF file
        try:
            files = list_repo_files(model_name)
            gguf_files = [f for f in files if f.endswith(".gguf")]

            if not gguf_files:
                raise ValueError(f"No GGUF files found in repository {model_name}")

            # If a specific filename was requested, use it
            if gguf_filename and gguf_filename in gguf_files:
                selected_file = gguf_filename
            else:
                # Prefer Q4_K_M or Q4_K_S for good balance of quality and speed
                # Fall back to any Q4 variant, then Q5, then Q8, then any available
                priority_order = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q8_0"]
                selected_file = None
                for priority in priority_order:
                    candidates = [f for f in gguf_files if priority in f]
                    if candidates:
                        selected_file = candidates[0]
                        break
                if not selected_file:
                    selected_file = gguf_files[0]  # Fall back to first available

            print(f"  Using GGUF file: {selected_file}")

            # Download the GGUF file
            gguf_path = hf_hub_download(repo_id=model_name, filename=selected_file)
            print(f"  GGUF file path: {gguf_path}")

            # Load tokenizer from the base model or a compatible tokenizer
            # Try to infer base model name from the GGUF repo name
            base_model_name = model_name.replace("-GGUF", "").replace("-gguf", "")
            if "/" in base_model_name:
                # For repos like unsloth/Qwen3-1.7B-GGUF, try to get the original model
                tokenizer_name = base_model_name
            else:
                tokenizer_name = model_name

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True,
                    padding_side="left"
                )
            except Exception as e:
                print(f"  Could not load tokenizer from {tokenizer_name}, trying {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model from GGUF file using from_single_file
            # Transformers 4.43+ supports loading GGUF files directly
            model = AutoModelForCausalLM.from_single_file(
                gguf_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map,
                trust_remote_code=True,
            )
            if device_map is None:
                model = model.to(_manual_device())

        except Exception as e:
            print(f"Error loading GGUF model: {e}")
            raise

    else:
        # Standard HuggingFace model loading
        # Configure quantization for T4 GPU (16GB VRAM)
        can_use_4bit = (
            use_4bit
            and torch.cuda.is_available()
            and accelerate_available
            and _package_available("bitsandbytes")
        )

        if can_use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            dtype = torch.float16
        else:
            if use_4bit and torch.cuda.is_available() and accelerate_available and not _package_available("bitsandbytes"):
                print("  bitsandbytes not found; loading fp16 instead of 4-bit.")
            quantization_config = None
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        if device_map is None:
            model = model.to(_manual_device())

    _model_cache[model_name] = model
    _tokenizer_cache[model_name] = tokenizer

    if torch.cuda.is_available():
        print(f"Model loaded! GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("Model loaded! (CPU mode)")
    return model, tokenizer


def call_model_huggingface(prompt: str, model_name: str = "Qwen/Qwen3-1.7B", max_new_tokens: int = 1024) -> str:
    """
    Call a HuggingFace model using local inference on T4 GPU.
    model: e.g., "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B", etc.
    """
    model, tokenizer = get_model_and_tokenizer(model_name)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response.strip()
