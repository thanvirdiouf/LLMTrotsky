# Decision after the controlled lower-LR test — 25 October 2026

## Source-grounding comparison complete — 25 October 2026

All **96 answers on24 fresh cases** completed on Kaggle without retraining. A separate clean-context coding agent scored randomized answers and locked scores before unblinding. Passages improve **19/24 base** and **22/24 saved1e-4** cases; critical failures fall **13→3** and **9→1**. Both grounded models retain12/14 positive controls at score≥3 and correct3/4 negative premises. Grounded adapter versus grounded base:7 improved/6 degraded/11 tied, with1 versus3 critical cases. **Prioritize grounding; use grounded base as the simpler reference, retaining the adapter as an experimental challenger.** Both still accept the Rakovsky proof premise despite explicit uncertainty; inspect this before any targeted training. Both adapters remain unchanged and experimental; no extra epoch, LR sweep or promotion. [Results](../evaluation/runs/kaggle_grounding_v1/QUALITY_REVIEW.md), [decision](../evaluation/runs/kaggle_grounding_v1/DECISION.md), [failure inspection](../evaluation/runs/kaggle_grounding_v1/FAILURE_INSPECTION.md). These24 are now development evidence; six source clusters and one coding-agent scorer limit generalization.


**Keep both revised adapters experimental. Do not add a full epoch or promote the1e-5 adapter.** The authorized controlled test is complete:50 actual updates on the same50 revised records/order/4407 supervised tokens and matching initialization,144 verified responses. Fresh1e-5 versus1e-4 gives5 improved/10 degraded/9 tied, with chronology improvement but calibration/detail losses and critical3/24 versus2/24. Positive pre-cutoff chronology means are unchanged. The hypothesis of preserving useful knowledge while retaining calibration gains is not supported overall by this small test.

The24 inspected v2 regressions separately give5 improved/4 degraded/15 tied and critical4/24 versus5/24. Both old base/control outputs exactly reproduce all24 responses/tokens; single-reviewer repeat score variation is disclosed. No next training, LR sweep, added epoch or deployment is selected or authorized. Any further experiment needs a new decision, fresh primary cases and stronger review of historical scoring. [Full matched evidence](../evaluation/runs/kaggle_lr1e5_v1/QUALITY_REVIEW.md) and `next_experiment_decision.json` govern the current decision.

## Recommended continuation — not started

1. Independently re-review the 24 inspected v3 cases using the saved answers and primary-source evidence, particularly chronology gains, calibration losses and borderline critical classifications. This is the immediate next recommendation and requires no GPU inference or training. Preserve the existing 144 scores, key and hash lock; put any new review in a separate directory, keep condition labels hidden during scoring, and disclose who actually reviewed it. A replay by the same reviewer does not establish independence. Compare disagreements explicitly rather than silently replacing this report.
2. After that measurement check, consider a source-grounding experiment on **new primary cases**: untouched base and the saved revised 1e-4 adapter, each without passages and with the same relevant passages (four matched conditions). Hold prompts outside the passage treatment, decoding and evidence selection fixed. Start with curated passages to isolate whether evidence helps; selecting/retrieving passages automatically would be another variable. Keep evaluation evidence out of training. The 1e-4 adapter is an experimental comparator, not a promoted model.
3. Choose any further training only after those results. No new review, source-grounding run, training run, epoch or deployment has been started by this recommendation. V1, v2 and v3 cases have all been inspected and are development/regression evidence for subsequent work, not fresh tests.

## Historical decisions before this completed test

# Next decision after revised-data v2 — 25 October 2026

The approved 50-update experiment and 72-response review are complete. **Retain the revised data as a useful diagnostic intervention, keep the adapter experimental, and do not add another epoch.** Compared with original-data step 50, revised step 50 improves 8/24 case scores, ties 16 and degrades none; critical cases fall 7 to 5. Two unsupported-detail refusals and all six explanations improve, but chronology/correction scores do not change, all three misleading premises remain critical, and every explanation scores below 3/4. See [full matched results and limitations](../evaluation/runs/kaggle_data_revision_v2_v1/QUALITY_REVIEW.md).

The next **proposed, not authorized** experiment is a fresh adapter with **lr1e-5 instead of lr1e-4**, retaining the frozen revised 50 records, exact order, 50 actual updates and all other settings. Compare untouched base, saved revised-data lr1e-4 step50 and the lower-rate adapter under identical decoding. This isolates the proposed optimizer change more narrowly than increasing epochs or changing data again; it does not establish that learning rate caused the present defects. A lower rate could lose the two observed gains. No new run has started.

Freeze new source-separated cases before that experiment. V2 is now development evidence and can serve as a regression set. Prioritize accurate versus misleading premises, documented versus fabricated visits, partial refusals that retain an unsupported event, real supplied quotations and source-specific explanation mechanisms. Do not train on these inspected questions, references or generated responses. Independent historian/human review remains absent. Full evidence and the machine-readable proposal are in `next_experiment_decision.json`.

## Historical selection before v2 preparation

# Selected next experiment: revise the data, keep a short fixed training budget

The fresh comparison does not demonstrate a useful adapter: step 50 already fails all six chronology cases and has five critical memory/citation failures; step 500 has six. Step 500 gains some false-premise correction but does not recover calibration or requested explanation. Neither stopping at 50 nor continuing the same recipe to 500 establishes reliability. The base is also poor.

**Selected change: the supervised data and behavior mix.** Start the next pilot from the untouched pinned base, with a new adapter and **50 actual optimizer updates**. Retain learning rate `1e-4`, rank 8, alpha 16, dropout 0, batch size 1, AdamW, constant schedule, masks, NF4/FP16 preparation and seed 42. Freeze the new data and exact order before running. Saved original-data step 50 is the diagnostic control. This first comparison changes the data intervention; do not simultaneously lower learning rate, change adapter targets, add retrieval or increase epochs and then attribute the result to one factor.

Prepare a separately versioned dataset; the original 500 and their locks remain historical controls:

1. Resolve the 21 revision proposals: make requested answers self-contained, correct inherited citation/context slips, put supported corrections or explanations in the turn where they are requested, and preserve honest date uncertainty.
2. Reconcile the 14 source-review examples; quarantine unresolved claims rather than silently choosing one conflicting edition/date. Version source/date/family adjudication with evidence.
3. Add reviewed examples of rejecting false premises, marking unknown future events, refusing unsupported private memories and bibliographic precision, and reasoning conditionally when the user explicitly supplies a hypothetical. Include true pre-cutoff facts, supported memories/quotations and answerable questions so blanket refusal is not rewarded.
4. Add source-grounded explanations that actually develop requested mechanisms and distinctions, alongside legitimate short answers. Do not impose a blanket answer-length target or write longer unsupported monologues.
5. Select and freeze a 50-update diagnostic slice that exercises these behaviors. Record its example IDs, per-answer token counts and loss exposure. Longer targets change token exposure even at fixed update count; report this as part of the data intervention.

All old and fresh evaluation families, prompts, evidence, generated answers and paraphrased answer targets stay excluded. The new holdout guard must pass. Do not generate training answers by copying or paraphrasing the inspected benchmark. Use separately reviewed non-held-out source material and independently written scenarios.

Before training, freeze a new source-separated evaluation with positive temporal controls, supported-versus-unsupported quotation/memory checks, misleading premises and multi-part explanations. Compare untouched base, saved original-data step 50 and the new-data step-50 adapter under identical decoding. The existing v1 evaluation is now development evidence and can provide secondary regression checks.

Primary checks are chronological boundaries, unsupported-memory/citation critical failures, direct premise correction and requested explanatory components. Check that gains are not blanket refusals and that fewer tokens are not omissions. A mechanically passing run, a lower training loss or zero truncations is insufficient. No promotion threshold is inferred from the poor base alone.

If calibration remains poor, the next separate optimization test should hold the revised data, order and 50-update budget fixed while lowering learning rate. The present trajectory does not identify a causal learning-rate effect; this is a follow-up hypothesis, not a demonstrated fix.

Historical selection status on 24 October: data revision and the next pilot had not started.

Current status, 25 October: the 526-record revised parent and 50-record slice are frozen; private Kaggle version 1 has been accepted and is running. The approved budget remains exactly 50 updates and 72 responses. The result will determine the next decision; no further run is authorized. See `data_revision_v2/WORK_LOG.md` and `data_revision_v2/status.json`.
