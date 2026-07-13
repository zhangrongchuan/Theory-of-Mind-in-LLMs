# Theory of Mind on HiToM and BigToM

本项目用于在 HiToM 和 BigToM 数据集上评测大语言模型的 Theory of
Mind（ToM）推理能力。HiToM 保留原有 method 实现；BigToM 通过独立适配器
转换为二选一 A/B 任务，并保持已有 final 实验的 prompt 和推理配置可复现。

支持的主要方法包括 `VP`、`SoO`、`SIMTOM`、`PercepToM`、`DWM`、
`DTOM`、`S3AP`、`INCREMENTALTOM`、`SHAREDEVIDENCETOM` 和
`assemableTom`。

## 1. 环境安装

建议创建独立 Python 环境，然后安装依赖：

```bash
pip install -r requirements-colab.txt
```

## 2. 数据集与通用命令

HiToM 默认读取 `data/hitom.json`：

```bash
python main.py --dataset hitom --category CoTP --method VP --max_samples 10
```

BigToM 默认读取 `data/bigtom_balanced_subset.json`：

```bash
python main.py --dataset bigtom --method VP --max_samples 10
```

推荐显式指定本地模型：

```bash
python main.py --dataset bigtom --method VP --model_name Qwen/Qwen3-1.7B
```

`--model_name` 和 `--qwen_model` 完全等价。BigToM 使用参考实验一致的
`model_hf.py` 推理配置，默认 `max_new_tokens=2048`；HiToM 保留原后端，
默认 `max_new_tokens=1024`。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--dataset hitom\|bigtom` | 选择数据集 |
| `--category CoTP` | HiToM 的 `prompting_type` 过滤条件；BigToM 通常不需要 |
| `--method METHOD` | 选择运行方法，具体参数值见下表 |
| `--model_name MODEL` | HuggingFace 模型，例如 `Qwen/Qwen3-0.6B` |
| `--max_samples N` | 只运行前 N 条；不设置则运行全部 |
| `--qwen_max_new_tokens N` | 覆盖默认最大生成长度 |
| `--chunk_size N` | IncrementalToM/assemableTom 的句子分块大小 |
| `--input_path PATH` | 覆盖默认数据文件 |
| `--output_path PATH` | 覆盖默认结果文件 |
| `--resume` | 从已有 JSONL 结果末尾继续 |
| `--upgrade` | 仅重新运行 `correct=0` 的样本 |

可选的 method 参数：

| 方法 | `--method` 参数值 | 简单说明 |
| --- | --- | --- |
| VP | `VP` | 直接读取故事和选项并回答，作为基础 baseline |
| SoO | `SoO` | 将模型置于目标人物的处境中进行推理 |
| SimToM | `SIMTOM` | 先筛选目标人物知道的事件，再从该视角回答 |
| PercepToM | `PercepToM` | 按“感知 → 信念 → 回答”三个阶段推理 |
| DWM | `DWM` | 构建分段的环境与人物信念状态描述 |
| Decompose-ToM | `DTOM` | 递归识别主体、重写问题并建立人物 world model |
| S3AP | `S3AP` | 先生成结构化 social-world representation |
| IncrementalToM | `INCREMENTALTOM` | 按句子块维护中间理解 checkpoint，可通过 `--chunk_size` 调整 |
| SharedEvidenceToM | `SHAREDEVIDENCETOM` | 提取相关人物共同知道的 shared epistemic evidence |
| AssembleToM | `assemableTom` | order 0–2 路由到 IncrementalToM，order 3–4 路由到 SharedEvidenceToM |

统一运行格式：

```bash
# HiToM
python main.py --dataset hitom --category CoTP --method METHOD --model_name Qwen/Qwen3-1.7B --max_samples 10

# BigToM
python main.py --dataset bigtom --method METHOD --model_name Qwen/Qwen3-1.7B --max_samples 10
```

补充说明：复现已有 BigToM `INCREMENTALTOM`/`assemableTom` final 结果时使用
`--chunk_size 9`；一般 HiToM 增量运行使用 `--chunk_size 3`。
