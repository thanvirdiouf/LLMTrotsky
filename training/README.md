# Completed lower-learning-rate test — 25 October 2026

## Source-grounding comparison complete — 25 October 2026

All **96 answers on24 fresh cases** completed on Kaggle without retraining. A separate clean-context coding agent scored randomized answers and locked scores before unblinding. Passages improve **19/24 base** and **22/24 saved1e-4** cases; critical failures fall **13→3** and **9→1**. Both grounded models retain12/14 positive controls at score≥3 and correct3/4 negative premises. Grounded adapter versus grounded base:7 improved/6 degraded/11 tied, with1 versus3 critical cases. **Prioritize grounding; use grounded base as the simpler reference, retaining the adapter as an experimental challenger.** Both still accept the Rakovsky proof premise despite explicit uncertainty; inspect this before any targeted training. Both adapters remain unchanged and experimental; no extra epoch, LR sweep or promotion. [Results](../evaluation/runs/kaggle_grounding_v1/QUALITY_REVIEW.md), [decision](../evaluation/runs/kaggle_grounding_v1/DECISION.md), [failure inspection](../evaluation/runs/kaggle_grounding_v1/FAILURE_INSPECTION.md). These24 are now development evidence; six source clusters and one coding-agent scorer limit generalization.


Exactly50 actual updates and144 responses completed and were verified. Fresh1e-5 versus saved revised1e-4:5 improved/10 degraded/9 tied; chronology improves but calibration/detail decline. Both adapters remain experimental, with no added epoch or next training. [Full result](../evaluation/runs/kaggle_lr1e5_v1/QUALITY_REVIEW.md), [active evidence/continuation](lr_1e5/CONTINUE_HERE.md). One coding-agent review; repeat scoring varied.

# Training preflight and small pilot

## Current result — 23 October 2026

The 500-example pilot **completed and was locally verified** in [Kaggle version 361474126](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-500-example-pilot?scriptVersionId=361474126). All 500 actual updates, frozen-base/finite-gradient checks, step-50 recovery restore, and final adapter reload checks passed. All 36 baseline/adapter pairs completed. Outputs and adapter are archived in `training/runs/kaggle_361474126/`; see `verification.json` and `QUALITY_REVIEW.md`. Baseline/adapter truncations: 13/2; mean output tokens: 407.1/56.2. The provisional unblinded inspection found serious fabricated memories/citations, false-date/premise acceptance and repetition. This adapter remains experimental; shorter outputs do not establish improvement. Independent blinded scores remain blank. Full recovery checkpoints remain on Kaggle; the compact local archive excludes their optimizer state.

Full progress and next actions: [progress log](../docs/PROGRESS_LOG.md). The older sections below retain experiment history; this current result supersedes pending-run statements.

The full one-epoch pilot notebook is now prepared: see [PILOT.md](PILOT.md), [frozen protocol](pilot_protocol.json), and [`notebooks/kaggle_qlora_pilot.ipynb`](../notebooks/kaggle_qlora_pilot.ipynb). It visits all 500 examples once, adds verified resumable checkpoints and a corrected-stop matched development comparison.

**Passed on Kaggle:** [version 361399211](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-training-smoke-test?scriptVersionId=361399211), 23 October 2026. All five compatibility checks passed: masks/end tokens/truncation, exact text adapters/frozen base, two finite optimizer updates, identical adapter tensors after a fresh-base reload, and matching greedy generated token IDs. Losses were 3.4608 and 4.2692; peak allocated GPU memory was 8.48 GiB (peak reserved 9.44 GiB). Only GPU 0 was used. The complete notebook ran in 346.2 seconds; the two training steps took 5.39 seconds, excluding loading and generation. This is a two-example compatibility result, not a 500-example training run or a quality improvement result. See [archived results](runs/kaggle_361399211/README.md).

The existing **500 agent-reviewed examples** in `datasets/train_seed.jsonl` are the complete training allowlist for this pilot. The notebook checks training compatibility on Kaggle: assistant answer token masks, complete answer terminators and truncation, text-only QLoRA targets, frozen base weights and absent/frozen vision parameters, finite loss and gradients, GPU memory, and adapter save/reload followed by generation. The working inference runs do not establish these properties. No GPU training result is claimed by the local checks.

`pilot_lock.json` freezes the dataset bytes and each reviewed conversation, current persona, review evidence, source-family assignments, evaluation inputs and existing benchmark artifacts. It pins `Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The recorded baseline uses NF4 double quantization, FP16 compute, non-thinking prompts, greedy decoding, seed 42 and a 512-token output ceiling. Its saved manifest also records all generation parameters and the exact concise instruction used by that comparison. Preserve the selected prompt condition when comparing an adapter; a different prompt condition is a separate experiment.

[`notebooks/kaggle_qlora_smoke.ipynb`](../notebooks/kaggle_qlora_smoke.ipynb) is ready to import into a fresh Kaggle notebook with Internet and GPU T4 x2 (GPU 0 only). It bundles all reviewed training rows and helpers, so no data attachment is needed. The default is two optimizer steps, batch size one, rank 8, alpha 16, learning rate 1e-4, no packing and max length 1,024. It targets all 248 verified text decoder Linear projections across both attention types and MLPs. The text-only loader omits the vision tower. PEFT preparation precedes adapter attachment. Frozen input/output matrices are staged on CPU during PEFT’s global FP32 cast, downcast there, then returned to GPU; every parameter must be on GPU before computation. Norms stay FP32. Actual package versions, exact target names and settings are recorded with the run.

The [real-tokenizer CPU preflight](token_mask_preflight.json) passed 500 examples (435 single-turn / 65 multi-turn), covering 565 answers and 23,792 supervised tokens. Input lengths are 251–457 tokens; no example is dropped at 1,024. See the [single-turn](token_mask_inspection_single.json) and [multi-turn](token_mask_inspection_multi.json) token tables. These checks loaded tokenizer files only, without model weights or a GPU. Reproduce with the pinned tokenizer libraries and Jinja2:

```bash
python scripts/verify_training_masks.py --tokenizer-path /PATH/TO/PINNED/TOKENIZER --max-length 1024
```

A Kaggle pass requires `run_manifest.json` status `passed` and all checks true. The folder also preserves per-step finite loss/gradient checks, memory, mask tables, update evidence, adapter tensors/configuration, baseline/trained/reloaded introduction probes and actual package pins. A fresh base must reload identical adapter weights and reproduce the trained greedy token IDs. The introduction probe uses 64 output tokens and is only operational; quality comparisons retain the locked 512-token baseline and exact selected prompt condition. Save a private version with outputs or download the entire run folder before ending the session. Failures retain their stage and partial artifacts. The exact stack passed the two-step smoke run linked above.

Validate the frozen inputs from the repository root:

```bash
python scripts/build_training_pilot.py
python -m unittest tests.test_training_pilot
```

After deliberately reviewing a changed input, explicitly rebuild the lock and notebook bundle:

```bash
python scripts/build_training_pilot.py --write-lock
python scripts/build_kaggle_training_smoke.py
```

`scripts/build_training_pilot.py` uses only the Python standard library and imports no model libraries. `build_lock(project)` checks all 500 rows against their approved `seed_records.jsonl` and existing review evidence lock, verifies catalog/source-split family assignments, and rejects evaluation/generated-answer copies. `validate_lock(project, lock)` rejects stale provenance, benchmark inventories or settings. A standalone notebook can bundle this helper and call `validate_training_payload(original_jsonl_bytes, lock)` before extracting messages. Only the reviewed JSONL enters training; source evidence, evaluation references, evaluation model inputs and generated benchmark answers are excluded. The portable lock contains exclusion fingerprints, never the reference passages or benchmark answer text.

Copy guards normalize Unicode, case and punctuation and reject exact excluded texts or shared contiguous 16-word spans. They do not prove absence of paraphrased contamination. Exact byte and per-example checks bind the notebook to the reviewed allowlist. The latest archived run includes 12 complete generated answers; older runs retain local inspection/review metadata rather than complete raw-answer archives. Those entire evaluation artifacts remain outside training through the allowlist, even where their full answer strings cannot be scanned locally.

Training settings, selected smoke examples, tokenization/template hashes, exact adapter module names, package versions, prompt/decoding settings, observed masks and truncation decisions, memory/gradient checks and save/reload results must be saved with the Kaggle run. Record the lock hash and training dataset checksum in that run manifest. The input lock alone does not constitute a frozen training run.

The six repeatedly inspected *Revolution Betrayed* questions are **development evidence**. Source-family separation does not make repeatedly inspected questions a fresh test, nor establish absence from base-model pretraining. Before broad conclusions or scaling, add fresh evaluation across other works and failure types. Expanding the training set, retrieval, and GPU utilization tuning can wait until this preflight and small pilot have been assessed.


The first Kaggle attempt, [version 361392774](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-training-smoke-test?scriptVersionId=361392774), failed before any optimizer step during PEFT’s temporary FP32 embedding/head allocation (requested 3.79 GiB with 3.44 GiB free). Its original reports and submitted notebook are preserved in `training/runs/kaggle_361392774/`. The revised preparation stages only these frozen matrices on CPU, then verifies that all base parameters return to the single GPU before computation. This preparation staging is not CPU compute offload.

The pinned tokenizer’s `<|im_end|>` is **248046**, while the checkpoint text config used by earlier baselines specifies **248044**. The operational save/reload probe now explicitly stops on 248046 and records the override. Earlier baseline decoding is preserved as historical evidence in the input lock. Before evaluating a trained pilot, freeze and run a **fresh matched baseline and adapter comparison** with the corrected stop token and otherwise identical prompts/decoding. Do not compare the corrected adapter’s quality directly against earlier answers with the different stop setting.


The second attempt, version 361396447, completed preparation and exact adapter/frozen-base checks at about 7.74 GB GPU allocation, then rejected non-finite gradients before optimizer step 1 with FP16 AMP’s default 65,536 loss scale. The next retry explicitly starts AMP loss scaling at 1.0, records it in settings, and keeps strict finite-gradient checks; no skipped update counts as success. Per-step AdamW counters must confirm each actual update. This remains a compatibility diagnostic until a full pass.
