# One-epoch 500-example Kaggle pilot

## Current result — 23 October 2026

The 500-example pilot **completed and was locally verified** in [Kaggle version 361474126](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-500-example-pilot?scriptVersionId=361474126). All 500 actual updates, frozen-base/finite-gradient checks, step-50 recovery restore, and final adapter reload checks passed. All 36 baseline/adapter pairs completed. Outputs and adapter are archived in `training/runs/kaggle_361474126/`; see `verification.json` and `QUALITY_REVIEW.md`. Baseline/adapter truncations: 13/2; mean output tokens: 407.1/56.2. The provisional unblinded inspection found serious fabricated memories/citations, false-date/premise acceptance and repetition. This adapter remains experimental; shorter outputs do not establish improvement. Independent blinded scores remain blank. Full recovery checkpoints remain on Kaggle; the compact local archive excludes their optimizer state.

Full progress and next actions: [progress log](../docs/PROGRESS_LOG.md). The older sections below retain experiment history; this current result supersedes pending-run statements.

Current private run: [Kaggle version 361474126](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-500-example-pilot?scriptVersionId=361474126). Status is recorded in `training/pilot_run_status.json`; submitted source is archived under `training/runs/kaggle_361474126/`.

`notebooks/kaggle_qlora_pilot.ipynb` is separate from the passing two-step smoke notebook. It starts a fresh adapter and performs exactly 500 verified optimizer updates: the longest 457-token example first, then a seed-42 shuffle of the remaining 499. Every reviewed example is visited once. No evaluation messages, passages or benchmark answers enter training.

The frozen protocol is `training/pilot_protocol.json`. It reuses Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, the passing package pins, NF4 double quantization, FP16 compute, FP32 adapter weights/norms, preparation-only CPU staging of frozen input/output matrices, explicit answer/end-token masks and AMP initial scale 1. Rank 8, alpha 16, dropout 0, AdamW learning rate 1e-4, no weight decay, constant learning rate, batch/accumulation 1, no packing, maximum length 1,024. All 500 examples must fit without drops.

Use a fresh Kaggle GPU T4 x2 session with Internet. GPU 0 holds the model; GPU 1 is not used for model computation. Dataset, model-input-only development cases and helpers are bundled. Save a private **Save & Run All** version with outputs. No GPU work runs locally.

Atomic checkpoints are published under `RUN_DIR/checkpoints/checkpoint-000000`, `checkpoint-000050`, and so on through `checkpoint-000500`. They contain adapter weights/config, AdamW state, AMP scaler, Python/NumPy/PyTorch CPU/CUDA random state, exact training order/identity, the next data index and coherent training metrics. Hash inventories and completion markers reject incomplete, changed or mixed state. The checkpoint at step 50 is actually restored in-place on Kaggle before continuing, to verify the restore path with real optimizer moments. Corrupt latest checkpoints never silently fall back.

To resume, attach your own saved complete run to a fresh Kaggle session, use the identical notebook/protocol/settings/package and GPU environment, and set `RESUME_CHECKPOINT` to a complete checkpoint folder. A new output directory is created; completed metrics are restored and only the remaining examples are trained. The final 500-step metrics must list all reviewed IDs exactly once. Full checkpoint state is needed for training continuation; the final adapter alone is for inference.

After training, the final adapter is saved and loaded on a fresh quantized base. Tensor equality and a 64-token greedy probe are checked. The notebook then generates all 36 existing development cases with the fresh base's adapter disabled and enabled, using identical original prompts, non-thinking rendering, greedy 512-token decoding and corrected `<|im_end|>` EOS 248046. `comparison_manifest.json` records the exact decoder configuration; `paired_answers.csv` provides an unblinded comparison. No automatic historical/persona quality scores are assigned. Historical baseline results using EOS 248044 are preserved separately.

Outputs include `run_manifest.json`, `training_order.json`, `training_metrics.jsonl`, `training_summary.json`, memory/mask reports, checkpoints, `final_adapter/`, matched evaluation journals, `evaluation_summary.json` and `paired_answers.csv`. A compact `pilot_delivery.zip` contains the final adapter, inputs, code and reports. Full resumable checkpoints remain in the saved Kaggle output tree and are deliberately excluded from the compact archive.

After downloading and verifying the complete compact archive, run `python3 scripts/export_pilot_review.py training/runs/kaggle_361474126`. It validates all 36 matched pairs, then writes condition-blinded review CSV/Markdown and a separate key under `review/`. Share only the review CSV/Markdown with reviewers until scoring is complete; the six rubric scores and critical-failure fields are intentionally blank. Reference evidence/checklists appear only in reviewer material. Existing scored review directories cannot be overwritten.


The 36 cases are previously inspected development evidence. Fresh evaluation across other works and failure types is needed before broad conclusions or scaling. A changing training loss or a completed run does not establish improved answer quality.

Rebuild/check locally without loading model weights:

```bash
python3 scripts/pilot_protocol.py
python3 scripts/build_kaggle_pilot.py
python3 -m unittest discover -s tests -q
```
