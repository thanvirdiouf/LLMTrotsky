# Audit of model advice and Kaggle notebook

## Current result — 23 October 2026

The 500-example pilot **completed and was locally verified** in [Kaggle version 361474126](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-500-example-pilot?scriptVersionId=361474126). All 500 actual updates, frozen-base/finite-gradient checks, step-50 recovery restore, and final adapter reload checks passed. All 36 baseline/adapter pairs completed. Outputs and adapter are archived in `training/runs/kaggle_361474126/`; see `verification.json` and `QUALITY_REVIEW.md`. Baseline/adapter truncations: 13/2; mean output tokens: 407.1/56.2. The provisional unblinded inspection found serious fabricated memories/citations, false-date/premise acceptance and repetition. This adapter remains experimental; shorter outputs do not establish improvement. Independent blinded scores remain blank. Full recovery checkpoints remain on Kaggle; the compact local archive excludes their optimizer state.

Full progress and next actions: [progress log](PROGRESS_LOG.md). The older sections below retain experiment history; this current result supersedes pending-run statements.

Checked on 22 October 2026. No model weights were downloaded, no inference was run, and no Kaggle session was launched during this audit.

## Model advice

All three named official checkpoints exist: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), and [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). They are post-trained models. Here, “baseline” means an untouched post-trained checkpoint with the project persona prompt, before project fine-tuning.

The earlier numerical ratings of 9B (6.5/10, 5.5–6/10) had no measured basis and are withdrawn. The project has generated no responses or quality scores. There is also no evidence that a seed of 500 examples favors the dense 27B model over the 35B MoE model. Those were hypotheses presented too confidently. Published general benchmarks do not determine historical-persona quality.

The MoE parameter description was correct: the official 35B-A3B card specifies 35B total and 3B activated parameters. The active count does not determine total weight storage. Approximate 16-bit weight storage is 18/54/70 decimal GB for 9B/27B/35B, excluding runtime memory. Exact fit and speed depend on precision, implementation, accelerator and context length. These are memory estimates, not measured Kaggle results.

Practical experiment: begin with 9B NF4 to verify the Kaggle path; compare larger candidates only on sufficient hardware, keeping quantization and decoding conditions consistent. Rank historical fidelity, chronology, attribution, voice and calibration from actual responses. Record critical failures separately. Use validation and behavioral cases for iteration; preserve test cases for final comparisons.

## Notebook findings and corrections

The original notebook passed syntax checks, but that was insufficient runtime verification:

- Its output separator was a literal backslash followed by `n`, rather than a newline. Multiple responses would not form valid JSONL, and resume/export would fail. Corrected and covered by actual round-trip tests.
- Loading 16-bit weights with no memory check or quantization path could overrun available Kaggle memory. The default now uses NF4, requires CUDA, checks rough GPU budgets, selects a supported compute dtype and disables CPU/disk offload. Fit is still subject to a real GPU smoke test.
- Resuming by case ID alone could reuse responses from different prompts/settings. Each run now records a resolved checkpoint commit, evaluation hash, template hash, effective generation configuration, quantization, dtype, seed, package versions and GPU identity. Outputs carry that run identity and prompt hashes.
- Token limits and completion status were not stored. Outputs now include input/output token counts and EOS-based `stop`/`length` status; oversized inputs fail instead of truncating silently.
- Dataset discovery arbitrarily selected the first matching filename. It now rejects ambiguous matches and verifies the frozen evaluation-file hash.
- The original human review export lacked the question/context and reference evidence. It now includes both, plus a separate model identity key and shuffled rows. No quality scores are fabricated. Keep scored copies separately: rerunning inference regenerates the blank review export.
- The non-thinking policy was correct. The revised notebook explicitly checks the rendered assistant prefix and tokenizes directly through the official chat template. `trust_remote_code=True` was unnecessary and is removed.

The earlier use of `AutoModelForCausalLM` is supported by the current Transformers [auto-model mappings](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/modeling_auto.py), which include Qwen3.5 full-config compatibility for the text-only classes. It was not established to be an error. The notebook checks that the installed version exposes the mapping before downloading weights. See also the [architecture documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [chat template guidance](https://huggingface.co/docs/transformers/en/chat_templating), and [bitsandbytes documentation](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).

The notebook uses an explicitly chosen greedy baseline. It is not Qwen's recommended sampled decoding for general non-thinking tasks. A sampled comparison can be a separate condition. Package versions are recorded at execution, rather than claimed as a previously GPU-tested stack.

## Validation evidence

- All 76 unit tests passed: 68 existing project tests plus 8 notebook tests. Notebook tests execute its helper functions and actual inference cell with a fake tokenizer/model runtime, including JSONL writing, resume and review export. They do not verify real GPU/model execution.
- `python scripts/validate_project.py` passed: 1,913 raw HTML checksums, 500 seed examples, 36 evaluation cases, and zero seed/evaluation source-family overlap. This verifies integrity and provenance consistency, not independent historical truth.
- Seed count checks agree with the documentation: 123 source documents, 111 families, 65 multi-turn examples, and 86 hearing examples in one source family; origin counts are 82 verbatim dialogue, 132 verbatim prose and 286 adaptations.
- Every non-install code cell compiles. Cells have valid notebook fields and unique IDs. The install cell is IPython syntax and must run in a notebook kernel.

Remaining verification: run the one-case smoke test on Kaggle, check model loading, actual peak memory, the rendered template, answer content, and `finish_reason`. Only then run all 36. No live Kaggle success or measured model quality is claimed.

## First Kaggle run: missing generation configuration

The user's [saved notebook, version 1](https://www.kaggle.com/code/thanvirdiouf/notebook4ba8d23bb7) exposes an actual error that the original fake runtime did not cover. GPU preflight detected two Tesla T4 devices, and the non-thinking prompt rendered correctly. The inference cell then raised `OSError` while loading `generation_config.json`, before model weights were loaded. Kaggle's overall run badge says successful, but the cell traceback establishes that inference did not succeed.

The resolved [checkpoint revision](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) has `config.json` but no separate generation configuration. The notebook now uses `GenerationConfig.from_model_config(config.get_text_config(decoder=True))`, deriving EOS/defaults from the already-loaded text configuration before applying the explicit benchmark settings. This uses the documented [GenerationConfig API](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig.from_model_config). Runner version is now 3. The inference regression test explicitly models the missing optional file, verifies use of the nested text configuration, and forbids the failing separate-file fetch.

All 76 offline tests pass after the fix. Live model loading and answer generation remain unverified; rerun the one-case smoke test after replacing that line in Kaggle.

## Successful one-case Kaggle smoke test

The user's [saved version 2, ID 361142169](https://www.kaggle.com/code/thanvirdiouf/notebook4ba8d23bb7?scriptVersionId=361142169), was inspected through its rendered notebook and output manifest. It includes the generation-config fix and successfully completes `memoir_books_ideas`: `1/36`, `finish_reason=stop`, zero length-limited responses. This establishes successful loading and one completed answer in that saved session; it is not a full evaluation. Kaggle's approximately ten-second Quick Version runtime should not be interpreted as model download or inference timing.

Observed manifest:

- Run ID: `7ae43a0305a77b6fc4b9786c340722793cd82d0d0b1e42b968182b534b39ce52`.
- Checkpoint: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Two Tesla T4 GPUs; float16 compute; bitsandbytes NF4 double quantization; non-thinking mode; seed 42.
- Packages: torch `2.11.0+cu128`, transformers `5.20.2`, accelerate `1.15.0`, bitsandbytes `0.50.2`, huggingface-hub `1.33.0`, tokenizers `0.23.1`.
- Saved runner identity is version 2 with the manual generation-config fix; the local corrected notebook labels this version 3.
- Output folder: `/kaggle/working/trotsky_baseline/Qwen3.5-9B-7ae43a0305a7`, with `responses.jsonl`, `run_manifest.json`, `pip_freeze.txt`, `human_review.csv`, and `review_key.csv` visible in the output tree.

The answer has a clear historical failure. It invents first-person recollections of growing up in the Caucasus and treats youthful absorption in nature/people as central. The reference in [My Life, chapter IV](https://www.marxists.org/archive/trotsky/1930/mylife/ch04.htm) says books and ideas occupied the greater place and that sensitivity to nature developed later; [chapter I](https://www.marxists.org/archive/trotsky/1930/mylife/ch01.htm) describes childhood in Yanovka. The response also attributes unsupported personal regrets to the persona. Record a source-faithfulness failure and a critical invented-memory flag. This is a qualitative agent review of one case, not independent human scoring or evidence for a global 9B rating.

The kernel-fallback messages explicitly say the reference PyTorch implementations are correct but slower. The ignored `top_k` warning is compatible with the selected greedy decoding. Neither prevented this completed answer. Keep this baseline's settings fixed and finish the 36 cases before changing its prompt or tuning on failures, especially because the smoke case belongs to the test split.

## Parallel execution update

At the user's request, the primary notebook now bundles `scripts/kaggle_parallel.py` and defaults to 36 cases with one independent process/model copy per GPU. The former serial notebook is retained as `kaggle_prompt_baseline_serial.ipynb`. Fresh subprocesses restrict visible CUDA devices before importing torch, pin the complete model explicitly to that worker's single GPU, use one CPU tensor thread each, and read one shared cached checkpoint snapshot. The existing NF4, batch-size-one, greedy, non-thinking and token-limit settings are retained.

The new execution mode has a separate manifest identity. Workers receive only case IDs, model-input messages and prompt checksums; reference evidence stays in the parent. Separate journals are merged with run/prompt validation and conflicting-duplicate rejection. Tests exercise concurrent real subprocess startup, round-robin assignment, completed-output resume, increasing case limits, worker failure/partial export, missing answers despite successful exit status, exclusive run locking, interrupted-process cleanup, recovery of a torn final line with a backup, and restoration from a saved parallel directory. Model execution itself is covered by a fake torch/Transformers runtime that checks explicit GPU placement, decoding options and the absent-generation-config fix. No local model weights or model inference are used in these tests.

Per-case synchronized generation timing and output-token rates, worker logs, package freeze and execution summaries support measuring the result on Kaggle. Actual parallel GPU execution, peak memory and speedup have not yet been verified. The design follows [Accelerate's independent-copy inference approach](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference) and the Hub's [snapshot caching API](https://huggingface.co/docs/huggingface_hub/en/guides/download); it does not assume that GPU utilization or speedup doubles.

The full suite passes: **88 tests**, including 20 notebook/parallel-runner tests. The bundled worker source matches the standalone script, and every non-install notebook code cell compiles after accounting for the `%%writefile` cell magic.

## First complete serial baseline

The user's [saved version 361150644](https://www.kaggle.com/code/thanvirdiouf/notebook4ba8d23bb7?scriptVersionId=361150644) contains all 36 serial responses: 23 normal finishes and 13 truncations at 512 tokens. The raw response document was parsed in the browser, and its case IDs were compared with the project's evaluation set. Observed metadata and preliminary failure findings are recorded in [the run inspection](../evaluation/runs/kaggle_361150644/README.md). This confirms complete response coverage for the initial token-budget condition, not successful historical behavior or a measured parallel speedup. Full rubric scoring remains outstanding.

## Verified parallel run and provisional rubric review

The user's [saved parallel version 361155394](https://www.kaggle.com/code/thanvirdiouf/notebook8b9f7bfe23?scriptVersionId=361155394) completed all 36 new cases, 18 per GPU. Its execution summary records 747.849 seconds plus a separately recorded 94.9 seconds of checkpoint preparation. All 36 raw response strings, prompt hashes, finish reasons and output-token counts were compared in the browser and match the serial baseline exactly. This verifies the parallel path on the actual two-T4 setup. Timing boundaries differ from the earlier serial run, so no exact benchmark speedup factor is claimed.

A [provisional coding-agent review](../evaluation/runs/kaggle_361155394/QUALITY_REVIEW.md) now scores all 36 cases against the project rubric, leaving inapplicable dimensions blank and keeping nine critical-failure cases separate from means. It is neither independent human nor blinded scoring. Thirteen answers remain truncated under the original 512-token condition. The review applies to both runs because their answer texts are identical; it does not restore an unsupported global model rating.

## Provided-source validation experiment prepared

At the user's request, [`kaggle_source_validation.ipynb`](../notebooks/kaggle_source_validation.ipynb) now bundles a paired experiment using only the six reserved validation questions. Twelve inputs compare exact original prompts with the same prompts plus verified historical source passages and a short instruction to use them. The persona system message, pinned 9B checkpoint, NF4, greedy non-thinking policy and 512-token limit are preserved. Checklists stay outside model input. The original full baseline notebook and shared worker are unchanged; the baseline builder now exposes a reusable notebook factory, and its generated artifact remains identical.

Source selection is oracle-based, not retrieval. All six questions come from one chapter of *The Revolution Betrayed*, which limits conclusions about general behavior. The new experiment has a separate output root and resume identity bound to model inputs, review references, settings and helper source. Neutral review IDs, equal review references and a separate condition key support condition-blinded scoring; answer prose may still reveal the condition. Unblinded paired answers and completion/truncation diagnostics are exported separately. The protocol and real-input preview are in [`evaluation/source_validation/`](../evaluation/source_validation/README.md).

All **97 unit tests passed**, including nine new tests for exact control prompts, validation-only selection, source/checklist separation, source checksums, balanced initial assignment, input-bound identity, blinded/partial review exports, reproducible notebook bundling, and a real CPU subprocess smoke/resume/full-export path. Synthetic subprocess answers stay in temporary test directories; they are not model results. Corpus validation again passed for 1,913 cached documents, 500 reviewed seeds and 36 evaluation cases, with no training/evaluation family overlap. No local model weights or inference were used. The new source-conditioned GPU run and its quality review remain pending on Kaggle.

The handoff snapshot generator no longer presents the evaluation builder's original zero-response count as current model-run status, or claims that remote model downloads have never happened after verified Kaggle inference.

## First provided-source GPU run inspected

[Saved version 361239681](https://www.kaggle.com/code/thanvirdiouf/notebook0554c058c2?scriptVersionId=361239681) completed all six validation pairs in 348.349 seconds plus 101.9 seconds of checkpoint preparation. All 12 raw responses were read in the browser. Six prompt-only and four provided-source answers hit the 512-token cap. The preliminary [paired inspection](../evaluation/runs/kaggle_361239681/README.md) finds useful source-specific corrections, but also a critical post-cutoff claim in the source-backed bureaucracy answer about judgments after Stalin's death. No broad reliability conclusion or automatic quality score is warranted.

The live run exposed a preflight token-count bug: the tokenizer returned a dictionary, so `len(encoded)` reported two fields. The builder now explicitly requests a token list with `return_dict=False`; a regression test checks actual length and over-limit rejection. Worker token checks already used the input tensor length correctly; actual recorded inputs were 247–803 tokens. The saved run remains usable, with no change to worker inference or prompts.


## Concise paired follow-up prepared

A provisional unblinded coding-agent review now covers all twelve responses from saved version 361239681. It assigns source-faithfulness means of 1.50/4 for prompt-only and 2.83/4 for supplied sources, with five improved pairs, but one identified critical failure in each arm. These are descriptive judgments of six related questions, not independent human scoring or a general model rating. See `evaluation/runs/kaggle_361239681/QUALITY_REVIEW.md` and `agent_review.json`.

The separate `notebooks/kaggle_source_validation_concise.ipynb` adds the identical 120-word instruction to both arms while keeping the original persona, questions, source passages, checkpoint, NF4, non-thinking greedy decoding and 512-token ceiling. Its identity and output root are separate. Word counts and limit compliance are exported independently from blank quality scores; no answer is trimmed or retried. The real-input preview and protocol are in `evaluation/source_validation_concise/`. Rebuilt helpers change their source checksum, so earlier saved notebooks remain the reference for resuming earlier runs.

All 103 unit tests passed, including checks of symmetric prompting, exact source preservation, old-prompt rejection, mixed-policy rejection, boundary word counts, partial exports, unchanged worker/decoding, notebook compilation and reproducible bundles. A second coding agent reviewed the implementation and found no blocker. No local model inference was performed.


## Concise GPU run completed and verified

After the user signed in to Kaggle, the agent created a separate private copy, imported the tested concise notebook, retained CasesForTest input, verified Internet and GPU T4 x2, and launched Save & Run All version 361242522. All six pairs completed with zero truncations and 1,870 output tokens. Execution including worker/model startup was 169.159 seconds, checkpoint preparation 100.6 seconds, and Kaggle total 340.2 seconds including setup/preflight/export. Shorter generation is not a new GPU-utilization benchmark.

Raw responses and manifest were archived locally and checked against all twelve expected prompts, the full manifest fingerprint, prepared-case identity and worker/helper source checksums. Answers are 112–141 words: one of twelve meets the exact 120-word instruction. No trimming or retries were performed. Two coding agents reviewed the raw new responses; the prior fabricated Mexican economic leadership and post-Stalin-death recollection do not recur, but unsupported personal self-revisions and conceptual mistakes persist. The review is unblinded and not independent historical scoring. Full verified results and screenshot are in `evaluation/runs/kaggle_361242522/`.
