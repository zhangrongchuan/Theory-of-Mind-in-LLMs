# Theory of Mind on HiToM and BigToM

This project evaluates the Theory of Mind (ToM) reasoning capabilities of large
language models on the HiToM and BigToM datasets. HiToM retains the original
method implementations, while BigToM uses dedicated adapters to convert its
examples into binary A/B tasks and preserve the prompts and inference settings
used in the existing final experiments for reproducibility.

The main supported methods are `VP`, `SoO`, `SIMTOM`, `PercepToM`, `DWM`,
`DTOM`, `S3AP`, `INCREMENTALTOM`, `SHAREDEVIDENCETOM`, and `assemableTom`.

## 1. Installation

We recommend creating a dedicated Python environment and then installing the
dependencies:

```bash
pip install -r requirements-colab.txt
```

## 2. Datasets and General Commands

By default, HiToM reads from `data/hitom.json`:

```bash
python main.py --dataset hitom --category CoTP --method VP --max_samples 10
```

By default, BigToM reads from `data/bigtom_balanced_subset.json`:

```bash
python main.py --dataset bigtom --method VP --max_samples 10
```

We recommend explicitly specifying the local model:

```bash
python main.py --dataset bigtom --method VP --model_name Qwen/Qwen3-1.7B
```

`--model_name` and `--qwen_model` are equivalent. BigToM uses the inference
configuration in `model_hf.py` that matches the reference experiments, with
`max_new_tokens=2048` by default. HiToM retains its original backend and uses
`max_new_tokens=1024` by default.

Common arguments:

| Argument | Description |
| --- | --- |
| `--dataset hitom\|bigtom` | Select the dataset |
| `--category CoTP` | Filter HiToM by `prompting_type`; generally not needed for BigToM |
| `--method METHOD` | Select a method using one of the values listed below |
| `--model_name MODEL` | Specify a Hugging Face model, such as `Qwen/Qwen3-0.6B` |
| `--max_samples N` | Run only the first N examples; omit it to run the full dataset |
| `--qwen_max_new_tokens N` | Override the default maximum generation length |
| `--chunk_size N` | Set the sentence chunk size for IncrementalToM/assemableTom |
| `--input_path PATH` | Override the default input data file |
| `--output_path PATH` | Override the default output file |
| `--resume` | Resume from the end of an existing JSONL result file |
| `--upgrade` | Rerun only examples with `correct=0` |

By default, result files are saved as
`res/<dataset>_<method>_<model>.jsonl`, for example
`res/bigtom_vp_qwen3_1_7b.jsonl`. If that file already exists, a new run uses
`_2`, `_3`, and so on instead of overwriting it. Without an explicit
`--output_path`, `--resume` and `--upgrade` automatically use the existing file
with the highest suffix.

Available method arguments:

| Method | `--method` value | Brief description |
| --- | --- | --- |
| VP | `VP` | Reads the story and answer choices directly and serves as the basic baseline |
| SoO | `SoO` | Reasons by placing the model in the target character's situation |
| SimToM | `SIMTOM` | First filters for events known by the target character, then answers from that perspective |
| PercepToM | `PercepToM` | Reasons in three stages: perception, belief, and answer |
| DWM | `DWM` | Builds segmented descriptions of the environment and characters' belief states |
| Decompose-ToM | `DTOM` | Recursively identifies agents, rewrites the question, and builds character world models |
| S3AP | `S3AP` | First generates a structured social-world representation |
| IncrementalToM | `INCREMENTALTOM` | Maintains intermediate understanding checkpoints across sentence chunks; configurable with `--chunk_size` |
| SharedEvidenceToM | `SHAREDEVIDENCETOM` | Extracts shared epistemic evidence known by the relevant characters |
| AssembleToM | `assemableTom` | Routes orders 0–2 to IncrementalToM and orders 3–4 to SharedEvidenceToM |

General command format:

```bash
# HiToM
python main.py --dataset hitom --category CoTP --method METHOD --model_name Qwen/Qwen3-1.7B --max_samples 10

# BigToM
python main.py --dataset bigtom --method METHOD --model_name Qwen/Qwen3-1.7B --max_samples 10
```

To reproduce the existing final BigToM results for
`INCREMENTALTOM`/`assemableTom`, use `--chunk_size 9`. For typical incremental
HiToM runs, use `--chunk_size 3`.
