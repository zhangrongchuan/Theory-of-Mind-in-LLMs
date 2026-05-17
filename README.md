# Theory of Mind on HiToM

This project evaluates Theory of Mind (ToM) reasoning on the HiToM dataset using prompt-based methods and DeepSeek models.

## Project Goal

- Run ToM evaluation on HiToM samples.
- Support multiple prompting methods ( eg: `VP` and `COTP`).
- Save per-sample predictions and compute final accuracy and accuracy by `question_order`.


## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install openai python-dotenv
```

3. Create `.env` and set your DeepSeek API key:

```env
deepseek_api=YOUR_API_KEY
```

## Run

```bash
python main.py --category [CATEGORY] --method [METHOD] [--max_samples [VALUE]]
```

If you want to switch method or output path, update arguments inside `main.py` (the `run_dataset(...)` call).

## Experimental Results

Dataset: HiToM

### Overall Comparison (One Row Per Run)

Use this table for quick model/method comparison. Add one new row for each completed run.

| Model | Method | Prompt Version | Final Accuracy | Correct/Total | Result File |
| --- | --- | --- | --- | --- | --- |
| deepseek-chat (DeepSeek-V3.2) | VP | v1 | 0.6242 | 749/1200 | experiment_results/deepseek_hitom_vp_full.jsonl |
| deepseek-chat (DeepSeek-V3.2) | COTP | v1 | 0.6800 | 816/1200 | experiment_results/deepseek_hitom_cotp_full.jsonl | 

### question_order Comparison (Wide Format)

Use this table for cross-model and cross-method comparison by order. Add one new row per completed run.

| Model | Method | Prompt Version | Order 0 | Order 1 | Order 2 | Order 3 | Order 4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat (DeepSeek-V3.2) | VP | v1 | 0.9417 (226/240) | 0.5917 (142/240) | 0.5417 (130/240) | 0.5417 (130/240) | 0.5042 (121/240) |
| deepseek-chat (DeepSeek-V3.2) | COTP | v1 | 0.9750 (234/240) | 0.7292 (175/240) | 0.6875 (165/240) | 0.5417 (130/240) | 0.4667 (112/240) |

Benchmarks for Presentation
| Model | Method | Order 0 | Order 1 | Order 2 | Order 3 | Order 4 |
| Qwen3-0.6B | SoO | 0.3710 (23/60) | 0.4333 (26/60) | 0.2623 (16/60) | 0.2500 (15/60) | 0.3333 (19/60)|
| Qwen3-1.7B | SoO | 0.6167 (37/60) | 0.5000 (30/60) | 0.2667 (16/60) | 0.2000 (12/60) | 0.2000 (12/60) |
| Qwen3-0.6B | PercepToM | 0.5738 (35/60) | 0.3684 (21/60) | 0.4426 (27/60) | 0.3281 (21/60) | 0.3158 (18/60) |
| Qwen3-1.7B | PercepToM | --- | --- | --- | --- | --- |


## Notes

- `VP` prompt requests a single option letter output (`A`-`O`).
- Output parsing in `utils.py` supports strict single-letter, `Answer: X`, and fallback letter extraction.
