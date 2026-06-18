# Theory of Mind on HiToM

This project evaluates Theory of Mind (ToM) reasoning on the HiToM dataset using prompt-based methods and DeepSeek models.

## Project Goal

- Run ToM evaluation on HiToM samples.
- Support multiple prompting methods (eg: `VP`, `COTP`, `S3AP`, `SIMTOM`,
  `DWM`, `INCREMENTALTOM`, `SHAREDEVIDENCETOM`, `assemableTom`, `PercepToM`,
  `SoO`, and `DTOM`).
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

Outputs are written directly under `res/` as JSONL, one row per completed
sample. The filename format is:

```text
<dataset>_<category>_<method>_<model>.jsonl
```

For example:

```text
res/hitom_cotp_assemabletom_Qwen_Qwen3_1_7B.jsonl
```

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

## S3AP / Social World Models

`S3AP` implements the static third-person method from *Social World Models*
(arXiv:2509.00559). For each sample, it first parses the story into a
query-independent S3AP social world representation, then answers the multiple
choice question using the original story plus that representation as extra
information.

Run example:

```bash
python main.py --category CoTP --method S3AP --max_samples 10
```

## SIMTOM

`SIMTOM` implements the two-stage perspective-taking method from *Think Twice:
Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities*
(arXiv:2311.10227). It first filters the story to events known by the target
character, then answers the original multiple-choice question using that
filtered perspective.

Run example:

```bash
python main.py --category CoTP --method SIMTOM --max_samples 10
```

## DWM

`DWM` implements the Discrete World Models prompting technique from *A Notion
of Complexity for Theory of Mind via Discrete World Models*
(arXiv:2406.11911). It splits a story into sequential chunks, asks the model to
write compact state descriptions after each chunk, then answers the original
question using those explicit world-state descriptions.

Run example:

```bash
python main.py --category CoTP --method DWM --max_samples 10
```

`DWM` uses 3 story splits by default.

## SHAREDEVIDENCETOM

`SHAREDEVIDENCETOM` replaces the older shared epistemic core name. It extracts
target-object evidence for every question order, including objective order-0
questions and shallow order-1/2 belief questions.

Run example:

```bash
python main.py --category CoTP --method SHAREDEVIDENCETOM --max_samples 10
```

## INCREMENTALTOM

`INCREMENTALTOM` runs the incremental chunk-based ToM method. Use
`--chunk_size` to control the sentence chunk size.

```bash
python main.py --category CoTP --method INCREMENTALTOM --chunk_size 3
```

## assemableTom

`assemableTom` is the routed combination method. Orders `0`, `1`, and `2` run
`INCREMENTALTOM`; orders `3` and `4` run `SHAREDEVIDENCETOM`. It runs directly
through `main.py`; no separate run or retry scripts are needed.

```bash
python main.py --category CoTP --method assemableTom --chunk_size 3
```

## Notes

- `VP` prompt requests a single option letter output (`A`-`O`).
- Output parsing in `utils.py` supports strict single-letter, `Answer: X`, and fallback letter extraction.
- `S3AP` writes the generated representation and parser prompt into each JSONL
  row as `s3ap_representation` and `s3ap_parser_prompt`.
- `SIMTOM` writes the filtered perspective and perspective-taking prompt into
  each JSONL row as `simtom_perspective` and `simtom_perspective_prompt`.
- `DWM` writes story chunks, state prompts, and generated state descriptions
  into each JSONL row as `dwm_chunks`, `dwm_state_prompts`, and
  `dwm_state_descriptions`.
