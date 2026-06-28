The changes are complete. Here's a summary:

**`model_hf.py`** - Updated to have callable functions like `model.py`:
- `call_model_hf(prompt, model_name, max_new_tokens)` - Standard helper prompt
- `call_model_hf_SoO(prompt, model_name, max_new_tokens)` - SoO specialized system prompt
- Both use a cached model so it's only loaded once
- The original standalone `main()` is preserved for direct script execution

**`main.py`** - Updated imports and SIMTOMYOU branch:
- Added imports: `from model_hf import call_model_hf, call_model_hf_SoO`
- SIMTOMYOU now uses `call_model_hf` instead of `call_model_ollama` (line 99)
- All other methods (PercepToM, DToM, SoO, S3AP, SIMTOM, DWM) still use `call_model_ollama`

**To run on the bw-cluster:**
```bash
# Load CUDA and Python
module load devel/cuda/12.4  # or your cluster's CUDA module
module load python/3.11

# Install dependencies
pip install transformers torch accelerate

# Run SIMTOMYOU method (will use GPU via HuggingFace)
python main.py --category CoTP --method SIMTOMYOU --max_samples 100
```

The model (`Qwen/Qwen3-1.7B`) will be downloaded on first run and cached for subsequent calls.

Done. Here's what was added:

**`.env`** (line 3):
```
HF_TOKEN=your_huggingface_token_here
```

**`model_hf.py`** - Added at the top:
- Imports for `os`, `load_dotenv`, and `login` from `huggingface_hub`
- Loads `.env` file automatically
- If `HF_TOKEN` is set and not the placeholder, logs into HuggingFace
- If placeholder, prints a message that no token is configured

**To use a real token:**
1. Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Replace `your_huggingface_token_here` in `.env` with your actual token

The code works **with or without** the token:
- **Without token**: Public models like `Qwen/Qwen3-1.7B` work fine
- **With token**: Needed for gated models (Llama-2, etc.) or to avoid download rate limits