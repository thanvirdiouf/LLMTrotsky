# Training plan and design decisions

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

The target is a late-life Trotsky persona. `config/persona.json` currently uses Mexico, 19 August 1940 as an adjustable working assumption. Local work remains CPU-only. Inference and prompt/source baselines have completed on Kaggle. The pinned tokenizer has now been checked locally without model weights; The two-step QLoRA compatibility preflight passed on Kaggle in version 361399211.

## RAG timing

Keep source IDs, URLs, dates, authorship and speaker labels now. Build retrieval after establishing a baseline; it does not require a completed fine-tune. Fine-tuning and retrieval are complementary experiments, not mandatory sequential stages.

Compare: original Qwen checkpoint with the persona prompt; the same with retrieval; LoRA with the same prompt; LoRA with retrieval. Retrieval should filter by historical availability and distinguish authored passages from editorial commentary. If training specifically for retrieval, include source-conditioned examples so the model learns to use evidence rather than simply attaching passages at deployment. Never assume retrieval guarantees correct answers.

## Fine-tuning method

Start with supervised conversational fine-tuning using **standard LoRA** on the official post-trained Qwen3.5-9B checkpoint. This is an experimental baseline, not a claim that LoRA must outperform all alternatives.

- **QLoRA:** useful when remote GPU memory is constrained. The frozen base weights are quantized while LoRA adapters are trained; it is not a separate conversational learning objective. Compare quality and runtime against non-quantized LoRA when resources allow.
- **rsLoRA:** changes rank-dependent scaling and is a reasonable later comparison, particularly if testing higher ranks.
- **DoRA:** adapts weight magnitude and direction. It can improve on LoRA in some settings, but brings implementation and resource tradeoffs; test only after the baseline and evaluator work.
- **Full fine-tuning:** an option with much larger compute/memory cost and a greater scope for changing general behavior. The current small, specialized dataset does not justify making it the first experiment.
- **Continued training on original prose:** a different objective from conversational SFT, and it can itself use LoRA. Compare it only if source familiarity or voice remains inadequate; follow with conversational SFT and check for degradation in dialogue.
- **Preference tuning:** consider later only if there are actual reviewed response preferences and recurring errors it can target. It does not replace source-grounded supervised examples.

The current **500-example reviewed seed**, including 65 multi-turn conversations, is enough for the initial small pilot after the training preflight passes. Dataset expansion, retrieval and GPU utilization tuning can wait. No useful adapter has yet been demonstrated. The repeatedly inspected six validation questions are development evidence; add fresh evaluation across other works and failure types before broad conclusions or scaling.

## Current training preflight

The [two-step Kaggle preflight passed](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-training-smoke-test?scriptVersionId=361399211) with finite losses/gradients, frozen base, exact text targets and successful adapter save/fresh-base reload/generation. Peak allocated memory was 8.48 GiB on GPU 0. This verifies the specific smoke configuration, not model quality or the memory required for every pilot example. Original artifacts are under `training/runs/kaggle_361399211/`.

[`notebooks/kaggle_qlora_smoke.ipynb`](../notebooks/kaggle_qlora_smoke.ipynb) bundles the reviewed dataset and a checksum/provenance/exclusion lock. It takes two optimizer steps on one Kaggle GPU, verifies the exact hybrid text decoder adapter targets, freezes the base, checks finite loss/gradients and memory, and saves/reloads the adapter onto a fresh quantized base before generating. It uses explicit labels in a PyTorch loop; no unverified `assistant_only_loss` template behavior is assumed. Text-only `AutoModelForCausalLM` loading omits vision modules, which the runtime reports as absent.

The [real-tokenizer CPU report](../training/token_mask_preflight.json) passes all 500 examples / 565 assistant answers, with lengths 251–457 tokens and no drops at 1,024. Assistant answers and `<|im_end|>` are supervised; system/user text, assistant headers, empty thinking wrappers and separators are ignored. All multi-turn assistant answers are checked. Overlength examples are dropped whole, with no partial answer or synthetic end token. Token tables and a reproduction CLI are under [`training/`](../training/README.md). This establishes the token path, not GPU training compatibility.

The runtime pins the successful inference checkpoint and stack plus candidate PEFT 0.21.2, prepares the quantized model before attaching adapters, and records actual versions and dtype. Frozen embeddings/head are temporarily staged on CPU during preparation, cast to compute precision there, and restored to GPU before any forward/backward; normalization parameters stay FP32. Only a passing Kaggle run establishes the memory/gradient/save-reload behavior. The operational introduction probe has a 64-token cap; pilot quality comparisons retain their own locked 512-token decoding and exact prompt condition.

## Remote implementation checks

Before committing a long GPU run:

1. Use a trainer version explicitly supporting the actual Qwen3.5 architecture and checkpoint. Inspect its language-model modules; do not copy an older Qwen/Llama adapter-target list blindly. Keep vision components frozen for this text-only project.
2. Render several messages through the official chat template. Choose a consistent thinking/non-thinking policy for training and serving; do not fabricate historical inner monologues as training reasoning traces.
3. Inspect tokenized examples and label masks, especially boundaries, EOS tokens, truncation, and multi-turn data. Confirm the loss covers the intended assistant targets and not the historical system instructions or source metadata.
4. Verify trainable parameters, dtype/quantization, a few optimizer steps, checkpoint save/load, and adapter inference on a tiny smoke run.
5. Evaluate the untouched baseline before training and keep generation settings fixed across comparisons. Then use a short pilot before scaling data, epochs or adapter rank.
6. Pin actual successfully tested package versions and checkpoint revisions in the remote environment. No exact working stack or memory budget is claimed yet.

Dataset quality, attribution, chronological framing, and evaluation come before tuning many hyperparameters. Use the evaluation failure categories to decide whether the next improvement belongs in parsing, retrieval, examples, prompts, or optimization.

## References checked during this project

- [Official Qwen3.5-9B checkpoint](https://huggingface.co/Qwen/Qwen3.5-9B).
- [TRL supervised fine-tuning](https://huggingface.co/docs/trl/en/sft_trainer).
- [PEFT quantized training](https://huggingface.co/docs/peft/en/developer_guides/quantization).
- [rsLoRA paper](https://arxiv.org/abs/2312.03732).
- [DoRA paper](https://arxiv.org/abs/2402.09353).
- [Retrieval-augmented generation paper](https://arxiv.org/abs/2005.11401).

Verify current compatibility on the chosen remote GPU environment before implementation.


The first Kaggle attempt, [version 361392774](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-training-smoke-test?scriptVersionId=361392774), failed before any optimizer step during PEFT’s temporary FP32 embedding/head allocation (requested 3.79 GiB with 3.44 GiB free). Its original reports and submitted notebook are preserved in `training/runs/kaggle_361392774/`. The revised preparation stages only these frozen matrices on CPU, then verifies that all base parameters return to the single GPU before computation. This preparation staging is not CPU compute offload.

The pinned tokenizer’s `<|im_end|>` is **248046**, while the checkpoint text config used by earlier baselines specifies **248044**. The operational save/reload probe now explicitly stops on 248046 and records the override. Earlier baseline decoding is preserved as historical evidence in the input lock. Before evaluating a trained pilot, freeze and run a **fresh matched baseline and adapter comparison** with the corrected stop token and otherwise identical prompts/decoding. Do not compare the corrected adapter’s quality directly against earlier answers with the different stop setting.


The second attempt, version 361396447, completed preparation and exact adapter/frozen-base checks at about 7.74 GB GPU allocation, then rejected non-finite gradients before optimizer step 1 with FP16 AMP’s default 65,536 loss scale. The next retry explicitly starts AMP loss scaling at 1.0, records it in settings, and keeps strict finite-gradient checks; no skipped update counts as success. Per-step AdamW counters must confirm each actual update. The third run with scale 1.0 passed both actual optimizer updates and the complete save/reload checks.
