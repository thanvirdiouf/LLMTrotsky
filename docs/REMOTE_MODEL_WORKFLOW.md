# Latest verified remote result —25 October2026

Controlled1e-5 test `thanvirdiouf/trotsky-revised-lr1e5-50-updates/1`:50 actual updates and144 matched responses complete;14 runtime checks and local archive/checkpoint/journal checks passed. Same revised50 examples/order/4407 tokens/initial adapter, only LR changed. Fresh results do not show an overall gain:5 improvements/10 degradations/9 ties against1e-4, with chronology up but calibration/detail down. Both adapters experimental; no added epoch or next run. [Full review](../evaluation/runs/kaggle_lr1e5_v1/QUALITY_REVIEW.md), [artifacts](../evaluation/runs/kaggle_lr1e5_v1/README.md). All model/GPU work stayed on Kaggle. Earlier sections retain workflow history.

# Remote generation and evaluation client

## Completed revised-data experiment — 25 October 2026

Private Kaggle version 1 completed **50 verified optimizer updates and 72 matched responses**. Archive, frozen protocol/data/helpers, token exposure, checkpoints, recovery/reload and response journals all passed local verification. One coding agent scored all 72 neutral answers before unblinding. Compared with original-data step 50, revised step 50 improves **8 cases, ties 16, degrades none**; critical cases fall **7/24 to 5/24**. Chronology and premise correction are unchanged, all three misleading premises remain critical, and every requested explanation remains below 3/4. The adapter stays experimental. The next proposal is lr1e-5 with the revised data/order/50-update budget and all other settings fixed; **no further run is authorized or launched**. See [full results and limits](../evaluation/runs/kaggle_data_revision_v2_v1/QUALITY_REVIEW.md), [verification](../evaluation/runs/kaggle_data_revision_v2_v1/verification.json) and [next decision](../training/NEXT_EXPERIMENT.md).

The supervised-token exposure was 4,407 versus 2,404 at equal update count. This single-reviewer, 24-case diagnostic does not isolate data content from exposure or identify learning rate as the cause. The proposed lower-rate test needs fresh frozen evaluation; v2 is development evidence.

The following dated sections preserve earlier experiment and decision history.

## Current result — 24 October 2026

The targeted 500-example audit and fresh inference comparison are complete. The audit flags **35 examples: 21 revisions and 14 source reviews**; original data and locks are preserved. The CLI-only Kaggle comparison generated all **72 answers on 24 frozen cases across six other works**, with verified checkpoint/input/decoding identities. A condition-blinded coding-agent review found **11/24 critical cases for the untouched base, 14/24 for step 50, and 12/24 for step 500**. Both adapters score zero on all six chronology cases; step 500 gains some false-premise correction but fails all six memory/citation calibration cases. These are provisional agent judgments, not independent historian validation or global model ratings.

Selected next change: **repair and rebalance supervised data, then test a fresh adapter at a fixed 50-update budget with the existing optimizer settings**. Data and optimization effects remain unresolved; another full epoch is not selected. The new data and next pilot have not started. See [full comparison and limitations](../evaluation/runs/kaggle_checkpoint_compare_v1/QUALITY_REVIEW.md), [audit](../reviews/targeted_audit_500/README.md), [next experiment](../training/NEXT_EXPERIMENT.md), and [Kaggle CLI setup](KAGGLE_CLI.md). The older sections below retain experiment history.

## Current result — 23 October 2026

The 500-example pilot **completed and was locally verified** in [Kaggle version 361474126](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-500-example-pilot?scriptVersionId=361474126). All 500 actual updates, frozen-base/finite-gradient checks, step-50 recovery restore, and final adapter reload checks passed. All 36 baseline/adapter pairs completed. Outputs and adapter are archived in `training/runs/kaggle_361474126/`; see `verification.json` and `QUALITY_REVIEW.md`. Baseline/adapter truncations: 13/2; mean output tokens: 407.1/56.2. The provisional unblinded inspection found serious fabricated memories/citations, false-date/premise acceptance and repetition. This adapter remains experimental; shorter outputs do not establish improvement. Independent blinded scores remain blank. Full recovery checkpoints remain on Kaggle; the compact local archive excludes their optimizer state.

Full progress and next actions: [progress log](PROGRESS_LOG.md). The older sections below retain experiment history; this current result supersedes pending-run statements.

The completed preflight is the [QLoRA training smoke-test notebook](../notebooks/kaggle_qlora_smoke.ipynb). It is standalone: import into a fresh Kaggle session with Internet and GPU T4 x2; only GPU 0 is used and no input attachment is needed. The notebook bundles the reviewed 500 examples and the pilot integrity lock. It inspects actual token labels before loading model weights, then runs two optimizer steps and checks adapter save/fresh-base reload/generation. See [training instructions and frozen comparison inputs](../training/README.md). All 500 examples passed the real-tokenizer CPU preflight; The [two-step GPU run passed](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-training-smoke-test?scriptVersionId=361399211) with finite gradients, 8.48 GiB peak allocated memory and exact fresh-base adapter reload/generation. The inference notebooks below are completed baseline/development evidence, not training compatibility checks.

The current follow-up is [`notebooks/kaggle_source_validation_concise.ipynb`](../notebooks/kaggle_source_validation_concise.ipynb). Import it into a fresh **GPU T4 x2** Kaggle session with Internet and the original `evaluation/cases.jsonl` attached, then Run All. It retains the source experiment's model and 512-token limit, and adds the identical 120-word instruction to both arms. Its separate output root is `/kaggle/working/trotsky_source_validation_concise`; earlier run folders are incompatible. See [the concise protocol](../evaluation/source_validation_concise/README.md) for output review and [the completed original paired review](../evaluation/runs/kaggle_361239681/QUALITY_REVIEW.md). The agent launched [saved concise version 361242522](https://www.kaggle.com/code/thanvirdiouf/trotsky-concise-source-validation?scriptVersionId=361242522), which completed all six pairs with zero truncations; see [verified results](../evaluation/runs/kaggle_361242522/README.md). Rebuild with `python scripts/build_kaggle_source_validation.py --concise`.

For the next experiment, import [`notebooks/kaggle_source_validation.ipynb`](../notebooks/kaggle_source_validation.ipynb) with the same original `evaluation/cases.jsonl` dataset. Use a fresh **GPU T4 x2** session with Internet enabled and Run All. The default generates 12 answers: six validation questions, each with its exact original prompt and with a supplied verified source passage. It retains the successful baseline's pinned checkpoint, NF4, non-thinking template, greedy decoding and 512-token limit. All helper code is bundled. `LIMIT=2` checks the first complete pair; restore 12 and rerun inference to finish. This original condition completed on Kaggle in version 361239681: all six pairs, with ten of twelve answers truncated. Its provisional review finds source corrections alongside continuing critical failures.

The output root is `/kaggle/working/trotsky_source_validation`. Save a version with outputs, or download the run folder. Check `comparison_summary.json` for six complete pairs, then score `validation_review.csv` with `validation_review_key.csv` kept separate. `paired_answers.csv` is an unblinded side-by-side export. The source passages are selected from validation references, so this is an **oracle/provided-source** experiment, not a test of retrieval. Checklists stay outside model input. See [the experiment protocol](../evaluation/source_validation/README.md) for controls, limitations and interpretation. Rebuild it with `python scripts/build_kaggle_source_validation.py` after modifying its helper or the shared worker.

For inference on Kaggle's GPUs, import [`notebooks/kaggle_prompt_baseline.ipynb`](../notebooks/kaggle_prompt_baseline.ipynb), attach the original `evaluation/cases.jsonl`, enable Internet and **GPU T4 x2**, and run its cells. It now defaults to **36 cases with one full Qwen3.5-9B NF4 copy per GPU**, using independent Python workers. Set `LIMIT = 2` for a short parallel check before increasing it to 36 and rerunning inference. Start a fresh session, or stop any earlier inference before launching the new runner.

The notebook bundles its worker code, so only the `.ipynb` and evaluation dataset need uploading. Both workers share one cached checkpoint snapshot. They keep batch size one and the serial baseline's generation settings; each selected GPU must have enough memory for a full copy. Checkpoint and package versions are pinned to those observed in the successful serial smoke run. The user verified the parallel run in [saved version 361155394](https://www.kaggle.com/code/thanvirdiouf/notebook8b9f7bfe23?scriptVersionId=361155394): 36 completed cases, 18 per GPU, in 747.849 seconds excluding 94.9 seconds of checkpoint preparation. All answer texts match the serial run exactly; an exact speedup factor lacks a serial timer with matching boundaries. See [the results and review](../evaluation/runs/kaggle_361155394/README.md).

Each worker writes a separate journal and log. The parent validates and merges responses in case order, preserves partial progress on failure, and terminates its workers on interruption. Resume skips saved cases and detects conflicting duplicates. Save/download the run folder before the session ends; `RESUME_DIR` can restore the same parallel run from an attached dataset. The parallel run has its own identity and preserves the earlier serial artifacts separately. Generation seconds and tokens per second are recorded per case; `execution_summary.json` measures execution wall time, including worker loading and excluding checkpoint download preparation.

To modify the implementation, edit `scripts/kaggle_parallel.py` and `scripts/build_kaggle_notebook.py`, then run `python scripts/build_kaggle_notebook.py` to rebuild the standalone notebook. The previous [`kaggle_prompt_baseline_serial.ipynb`](../notebooks/kaggle_prompt_baseline_serial.ipynb) retains model sharding for configurations where a larger checkpoint needs several GPUs to hold one copy.

The user completed all 36 cases with the earlier serial runner in [saved version 361150644](https://www.kaggle.com/code/thanvirdiouf/notebook4ba8d23bb7?scriptVersionId=361150644). Raw records show 23 normal finishes and 13 length-limited answers. The subsequent provisional agent review applies to both runs because their answer texts match; independent human scoring remains outstanding. This saved version does not exercise the parallel code. See [the run inspection](../evaluation/runs/kaggle_361150644/README.md) and [the audit](RECENT_WORK_AUDIT.md) for details and verification limits.

`scripts/remote_model.py` is an HTTP client. It does not load a model, allocate GPU memory, or start a server. Use it with an already configured remote model service implementing a compatible chat-completions endpoint. No endpoint, model service, or credentials have been selected in this project, and no inference requests have been sent.

The client uses the HTTP chat format supported by [vLLM's compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/). Other servers/providers must be checked for compatibility. Package versions, model support and chat-template behavior still need a live remote smoke test.

## Prepare requests without inference

```bash
python3 scripts/remote_model.py generate --model TEACHER_MODEL --limit 1 --dry-run --output datasets/generation_requests.preview.jsonl
python3 scripts/remote_model.py evaluate --model QWEN_BASELINE --limit 1 --dry-run --output evaluation/requests.preview.jsonl
```

These previews contain actual prompts and source material but make no model calls. `TEACHER_MODEL` and `QWEN_BASELINE` are labels for previews, not downloaded models or chosen services.

## Call a configured remote service

Pass the model name served by that service and its API root, including `/v1` when required. Set `TROTSKY_MODEL_API_KEY` in the environment if the service requires authentication; the client does not write it into output files.

```bash
python3 scripts/remote_model.py generate --model YOUR_SERVED_TEACHER --base-url https://YOUR_SERVER/v1 --limit 1 --output datasets/teacher_outputs.jsonl
python3 scripts/remote_model.py evaluate --model YOUR_SERVED_BASELINE --base-url https://YOUR_SERVER/v1 --limit 1 --output evaluation/baseline_responses.jsonl
```

Start with one request and inspect the result. Explicit limits bound requests per invocation and generated output length. Successfully completed request signatures are skipped when the same output is resumed. There are no automatic HTTP retries; a server-side timeout may require checking the remote job before retrying. The output has an exclusive file lock against concurrent writers.

For Qwen served with a compatible chat-template implementation, `--request-options config/qwen_non_thinking_request_options.json` supplies the non-thinking template option. Verify that the chosen server/model supports it; do not use it blindly with an unrelated teacher. Keep the same thinking policy across baseline and adapter evaluation.

## Generation gates and outputs

By default generation reads the reviewed seed passages. A custom `--input` may select or reorder passages from the frozen seed review. A Boolean approval field alone cannot authorize a new passage: its source hash, block IDs, evidence checksum and date evidence must match an entry in `reviews/seed_evidence_lock.json`. Expand coverage by reviewing and rebuilding the seed first. The runner checks the cached HTML hash, source URL, block text and roles, authoritative composition evidence, current catalog family, split assignment, current evaluation-specification reservations and frozen passage membership. It also rejects stale review specifications or persona settings, unassigned/held-out families and sources outside the date boundary. Reusing the seed cannot create new historical coverage; expanding coverage requires more reviewed source passages first.

The teacher must return JSON with an `examples` array. Mechanical checks enforce nonempty questions/answers, valid origin labels, exact supporting spans, provenance, count limits, and verbatim matching when claimed. **Every generated candidate stays pending review and unapproved for training**, even if the teacher emits an approval flag. Adapted claims require a further source review; matching an evidence string does not prove all claims are entailed.

Evaluation sends only each case's model-input messages. Reference evidence and scoring checklists are withheld from that request. Responses, finish reasons and token usage are recorded; no automatic model-quality scores are invented. Use the scoring protocol in `evaluation/README.md`.
