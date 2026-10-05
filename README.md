# Qwen2.5-VL DPO Hallucination Mitigation Pipeline

This repository contains a reproducible pipeline for constructing multimodal DPO preference pairs from on-policy VQA answers, fine-tuning `Qwen/Qwen2.5-VL-3B-Instruct` with LLaMA-Factory LoRA DPO, and evaluating hallucination mitigation on AMBER.

It was built for a three-student team project in the course *Introduction to Multimodal Learning* at Sichuan University (2026). Hanwen Zhang wrote the preference-data pipeline (judging, re-judging, auditing) and the LoRA DPO training code in this repository; teammates ran the final training and AMBER evaluation.

## Final Results

The final run used 10,000 DPO pairs: 6,880 accepted from the first judge pass and 3,120 revised by re-judging low-confidence or small-gap cases (0 schema errors). The model was trained with LoRA DPO for 2 epochs and compared with the base model on the same AMBER queries (1,004 generative, 14,216 discriminative) with unchanged evaluation code.

| Metric | Base | LoRA DPO |
|---|---:|---:|
| CHAIR (lower is better) | 8.1 | 5.7 |
| Hal (lower is better) | 49.4 | 29.2 |
| Cog (lower is better) | 5.2 | 2.3 |
| Cover (higher is better) | 69.3 | 62.9 |
| Discriminative accuracy | 55.2 | 69.6 |
| Discriminative F1 | 57.4 | 74.8 |

Hallucination dropped on every generative metric, and discriminative F1 rose mainly through higher recall (41.6 → 62.6). Cover also fell, so the fine-tuned model answers more conservatively.

## Project Summary

The course data provides 10 candidate answers for each multimodal question. The core task is to turn each group of candidates into a DPO pair:

```json
{
  "instruction": "<image>{prompt}",
  "input": "",
  "chosen": "least-hallucinated answer",
  "rejected": "most-hallucinated answer",
  "images": ["images/xxx.jpg"]
}
```

This repo adds:

- OpenAI-compatible multimodal judge pipeline for DPO pair construction.
- Auditable preference metadata: confidence, score gap, candidate scores, reasoning, judge/refine model tags.
- `pixi` environment/tasks for data building, LLaMA-Factory registration, training, LoRA fixup, AMBER evaluation, and result summarization.
- LLaMA-Factory DPO profiles: a local RTX 4070S profile (4-bit QLoRA, for testing the pipeline) and the full-run profile (bf16 LoRA, as used for the final results).
- Sample preview files safe to commit.

## Important Data and Secret Policy

Do **not** commit secrets, generated datasets, model weights, or large provided data unless you have explicit permission to redistribute them.

Ignored by default:

- `.env`
- `data/raw/answers.jsonl`
- `images/`
- generated DPO/audit artifacts under `data/processed/` and `data/audit/`
- model checkpoints/weights/caches
- heavy AMBER outputs

Commit-safe examples live under:

```text
data/samples/
```

Copy `.env.example` to `.env` and fill your own OpenAI-compatible endpoint credentials:

```bash
cp .env.example .env
```

## Environment

This project uses `pixi`.

```bash
pixi install
pixi run compile-scripts
pixi run build-dpo-preview
```

The preview command uses a deterministic mock judge and does not call any external API.

## DPO Pair Construction

Main script:

```bash
scripts/build_dpo_pairs.py
```

Example real run:

```bash
pixi run python3 scripts/build_dpo_pairs.py \
  --answers answers.jsonl \
  --output data/processed/qwen2_5_vl_dpo_local.json \
  --audit data/audit/qwen2_5_vl_dpo_local.audit.jsonl \
  --judge-model gpt-5.4 \
  --refine-model gpt-5.5 \
  --enable-refine \
  --max-accepted 1000 \
  --max-questions 3000 \
  --confidence-threshold 0.70 \
  --refine-threshold 0.80 \
  --min-score-gap 1.0 \
  --workers 4 \
  --max-image-side 1024 \
  --image-jpeg-quality 85
```

Useful features:

- `--resume` continues from an existing audit/output pair.
- `--workers` enables parallel question-level API calls.
- `--max-image-side` and `--image-jpeg-quality` shrink uploaded images to avoid request-size errors.
- `--judge-mode mock` validates schema without spending API calls.

## Development Run (superseded)

During development, a smaller 5,059-pair dataset was built locally to test the pipeline. It is kept only as ignored local artifacts; the final results above use the 10,000-pair dataset.

```text
data/processed/qwen2_5_vl_dpo_5059_54mini_54refine_recovered.json
data/processed/qwen2_5_vl_dpo_5059_54mini_54refine_recovered_with_meta.jsonl
data/audit/qwen2_5_vl_dpo_5059_54mini_54refine_recovered.audit.jsonl
data/audit/qwen2_5_vl_dpo_5059_54mini_54refine_recovered.summary.json
```

Summary of that local run:

- Total DPO pairs: 5,059
- Schema errors: 0
- Average confidence: 0.924364
- Median confidence: 0.93
- Average score gap: 6.812143
- Median score gap: 7.4

These files are intentionally ignored by git. Regenerate them with your own credentials if needed.

## LLaMA-Factory Integration

Register the generated dataset into a local LLaMA-Factory checkout:

```bash
pixi run register-dpo-local
```

Training configs:

```text
configs/llamafactory/train_dpo_qwen2_5_vl_local.yaml
configs/llamafactory/train_dpo_qwen2_5_vl_full.yaml
```

Both preserve the course constraints:

- `stage: dpo`
- `finetuning_type: lora`
- `pref_loss: sigmoid`
- `template: qwen2_vl`
- `model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct`

## AMBER Evaluation

Use wrappers around the existing AMBER scripts rather than changing evaluation semantics:

```bash
pixi run eval-base-local
pixi run eval-lora-local
pixi run summarize-results
```

Full 8×A100 profile tasks are also provided:

```bash
pixi run eval-base-full
pixi run eval-lora-full
```

## Repository Structure

```text
configs/judge/                 # Judge defaults
configs/llamafactory/          # Dataset registration and train profiles
scripts/build_dpo_pairs.py     # DPO pair construction
scripts/register_llamafactory_dataset.py
scripts/fix_lora_adapter.py
scripts/eval_amber.sh
scripts/summarize_amber_results.py
data/samples/                  # Small commit-safe preview artifacts
```
