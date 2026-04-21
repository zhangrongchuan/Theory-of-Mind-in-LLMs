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
python main.py
```

If you want to switch method or output path, update arguments inside `main.py` (the `run_dataset(...)` call).

## Experimental Results

Dataset: HiToM

### Overall Comparison (One Row Per Run)

Use this table for quick model/method comparison. Add one new row for each completed run.

| Model | Method | Prompt Version | Final Accuracy | Correct/Total | Result File | Run Date | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat (DeepSeek-V3.2) | VP | v1 | 0.6242 | 749/1200 | results/hitom_vp_results.jsonl | 2026-04-22 | Current baseline |

### Accuracy by question_order (Long Format)

Use this long-format table for detailed analysis. For each run, append 5 rows (order 0 to 4).

| Model | Method | Prompt Version | Question Order | Correct/Total | Accuracy |
| --- | --- | --- | --- | --- | --- |
| deepseek-chat (DeepSeek-V3.2) | VP | v1 | 0 | 226/240 | 0.9417 |
| deepseek-chat (DeepSeek-V3.2) | VP | v1 | 1 | 142/240 | 0.5917 |
| deepseek-chat (DeepSeek-V3.2) | VP | v1 | 2 | 130/240 | 0.5417 |
| deepseek-chat (DeepSeek-V3.2) | VP | v1 | 3 | 130/240 | 0.5417 |
| deepseek-chat (DeepSeek-V3.2) | VP | v1 | 4 | 121/240 | 0.5042 |

### Recommended Naming Convention

- Model: exact API/model name, for example deepseek-chat, gpt-4.1, qwen2.5-72b-instruct.
- Method: VP, COTP, or other method name.
- Prompt Version: v1, v2, v3 (increase whenever prompt text changes).
- Result File: keep one jsonl file per run, for example results/hitom_deepseek_vp_v1.jsonl.

## Notes

- `VP` prompt requests a single option letter output (`A`-`O`).
- Output parsing in `utils.py` supports strict single-letter, `Answer: X`, and fallback letter extraction.
