# Project progress log

Updated 2026-10-23 20:11 IST. This log links current evidence; `PROJECT_HANDOFF.md` carries project context and `docs/RECENT_WORK_AUDIT.md` retains earlier audits.

## Documentation practice

Record each meaningful milestone, failure, fix, check and decision here and update the current handoff before ending a work session. Preserve exact submitted notebooks, run IDs/URLs, frozen settings/input hashes and original results. Distinguish CPU checks, GPU execution, qualitative agent inspection and independent human review. Do not leave summary pages saying “pending” after a verified completion.

## 23 October 2026 — training preflight

- CPU tokenization: 500 examples, 565 assistant answers, 435 single-turn and 65 multi-turn; 251–457 input tokens and 23,792 supervised tokens. Zero drops at max length 1,024. Explicit assistant-answer plus `<|im_end|>` masks; other tokens ignored.
- Kaggle 361392774: OOM during PEFT preparation. Fixed transient FP32 embedding/head allocation by staging the frozen matrices on CPU during preparation, then restoring them to GPU before compute.
- Kaggle 361396447: preparation fit, but nonfinite gradients with default AMP scaling. Explicit initial loss scale 1.0 fixed the passing configuration.
- Kaggle 361399211: two actual training updates passed; peak allocation 8.48 GiB, frozen base, 248 text projections, exact final adapter reload and matching greedy tokens. This established compatibility, not quality.
- Corrected answer EOS is 248046; old baseline setting 248044 remains preserved in historical artifacts. The pilot generates a fresh corrected-stop baseline.

## 23 October 2026 — one-epoch pilot

- Frozen model: Qwen/Qwen3.5-9B at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Frozen dataset SHA-256: `b2701476c7dc8e9f0da811efb0417eb380af64e53666cdb526bd6500cc68f5f1`.
- Protocol/settings: `training/pilot_protocol.json`; rank 8 / alpha 16, NF4 double quantization, FP16 compute, FP32 adapters/norms, AdamW lr 1e-4, batch/accumulation 1, 500 updates. Longest example first, seeded shuffle thereafter, each example once.
- Kaggle 361472632: stopped before model loading or training because `BitsAndBytesConfig` was not imported. Submitted source and failure evidence archived. Fixed import and added bundled-helper undefined-name checks; 156 core CPU tests passed. Eight later review-export tests also passed.
- Kaggle 361474126: completed in 50m40s overall; training updates took about 17m23s. Checkpoints 0 and every 50 updates through 500 saved. Real adapter/optimizer/scaler/RNG restore at 50 passed. Final fresh-base adapter tensors and probe token IDs matched.
- Downloaded 85.9 MB compact archive through the authenticated Kaggle browser. Checked ZIP CRC/path safety, helper/dataset/protocol hashes, all 500 unique ordered actual optimizer counters, finite losses/gradient norms, frozen gradients and probe IDs. Strict review exporter validated all 36 matched prompt/decoding/token-stop pairs. `verification.json` stores file hashes and verification scope.
- Training mean loss 2.5706; first/last 25 means 2.8298/2.4983. These are training diagnostics, not quality scores.
- The 500-example pilot **completed and was locally verified** in [Kaggle version 361474126](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-500-example-pilot?scriptVersionId=361474126). All 500 actual updates, frozen-base/finite-gradient checks, step-50 recovery restore, and final adapter reload checks passed. All 36 baseline/adapter pairs completed. Outputs and adapter are archived in `training/runs/kaggle_361474126/`; see `verification.json` and `QUALITY_REVIEW.md`. Baseline/adapter truncations: 13/2; mean output tokens: 407.1/56.2. The provisional unblinded inspection found serious fabricated memories/citations, false-date/premise acceptance and repetition. This adapter remains experimental; shorter outputs do not establish improvement. Independent blinded scores remain blank. Full recovery checkpoints remain on Kaggle; the compact local archive excludes their optimizer state.
- Prepared condition-blinded review CSV/Markdown with a separate key. Current quality inspection is explicitly provisional and unblinded; no fabricated numeric quality scores.

## Operational preferences

- GPU work stays remote. No model inference or training runs on this PC.
- Do not continuously poll a long Kaggle run. Let it run independently and inspect when the user reports completion or requests status.
- Prefer terminal/CLI downloads when authenticated access is configured. Kaggle CLI/package and credentials were absent when checked; current download is complete. Provide setup instructions when next needed. Never store credentials in this repository or progress log.

## Next work — proposed, not executed

1. Audit the frozen 500-example training set against the observed failure types: temporal grounding, false-premise correction, unsupported private memories/quotes, answer detail, and repeated or context-dependent answers. Record example IDs, source evidence and proposed corrections separately; preserve the original dataset and pilot lock. Do not copy benchmark answers into training.
2. Define a fresh, source-family-separated evaluation set covering other works and the same failure types. Freeze prompts, rubric, stop settings and exclusions before another comparison. Existing 36 cases are development evidence.
3. After the audit, evaluate an earlier saved checkpoint (initially step 50 or 100) against the untouched base under identical settings. This checks whether failures emerge with update count before spending another full epoch. It is a proposed experiment, not yet run, and cannot alone prove the cause.
4. Use the audit and checkpoint comparison to choose a narrowly changed next pilot. Do not expand the dataset, add retrieval, change model size or tune GPU utilization merely to respond to this failed quality result. Keep all model/GPU execution on Kaggle.


## Evidence index

- `training/runs/kaggle_361474126/README.md`: completed run and archive boundaries.
- `training/runs/kaggle_361474126/verification.json`: integrity and per-file checksums.
- `training/runs/kaggle_361474126/QUALITY_REVIEW.md`: concrete failure examples and limits.
- `training/runs/kaggle_361474126/final_adapter/`: experimental adapter/tokenizer.
- `training/runs/kaggle_361474126/review/`: blank blinded reviewer materials and separate key.
- `training/pilot_run_status.json`: machine-readable final status.

## 24 October 2026 — complete targeted audit, frozen fresh cases, CLI inference submitted

Audited all 500 frozen examples / 565 answers: 465 retain, 21 revise, 14 source review, with 38 findings across 35 examples. Preserved canonical data and locks; proposed actions remain separate. Froze 24 fresh cases across 6 other works after independent coding-agent source/context and duplicate review; added source exclusion guards to future authoring/staging paths. Historical lock now correctly rejects the expanded live evaluation inventory; archived snapshot reproduction remains tested. Installed/authenticated official Kaggle CLI, hash-verified downloaded step 50/100 adapters, and submitted private inference-only version 1 comparing untouched base, step 50 and step 500 on identical fresh prompts. Remote result and training decision pending. No new epoch started.

## 24 October 2026 — fresh comparison completed and next change selected

Private CLI comparison version 1 completed all 72 answers: base/step 50/step 500 on 24 frozen cases across 6 other works. Archive, hashes, manifests, package/model/checkpoint identities, exact prompt/token/decoding journals and adapter files passed verification. Three coding agents scored neutral answer IDs before mapping was revealed: critical cases 11/24,14/24,12/24 respectively. Both adapters failed all 6 chronology cases; step 500 gained some false-premise correction but failed all 6 memory/citation cases. Scores remain provisional, not independent historian validation. Selected data revision/behavior coverage before a fixed 50-update fresh-adapter pilot with existing optimizer settings; next data and pilot not started. Original training and locks preserved; no extra epoch or local GPU work. 192 CPU tests and corpus/fresh-evaluation/score-aggregation integrity checks pass.


## 25 October 2026 — revised-data pilot accepted and running

Completed 21 evidence-bound repairs, quarantined 14 unresolved originals and added 40 reviewed examples: 526-record parent, 50-record slice, all 526 exact-tokenizer masks passing. Frozen source-separated v2 has 24 cases across 6 new works. Original data and locks preserved. All 236 preparation CPU tests passed. User explicitly approved private Kaggle upload, exactly 50 actual updates and 72 comparison responses. SaveKernel first returned HTTP 400; response-only diagnostics identified its 1 MB source limit. Archived failed source/state and repackaged all 13 byte-identical bundle payloads with LZMA/base85 (988,557 CLI-normalized bytes). Official CLI accepted private version 1; session is running, without yet verified optimizer completion. One coding agent will score all three neutral shards; the summary tool now records actual reviewer count and passes 18 targeted tests. No extra run is authorized.


## 25 October 2026 — revised-data run verified and condition-hidden review complete

Private Kaggle v2 version1 completed exactly 50 actual AdamW updates and 72 matched answers. Downloaded the 86,985,115-byte delivery ZIP, verified CRC/unique safe paths/all 40 extracted entries, frozen payload/helper/settings/package/model identities, exact order and exposure, all counters/finite gradients, checkpoint0/50 provenance, actual recovery restore, fresh reload/probe and all journals. Original/revised initial adapter file hashes match; five historical originals remain unchanged. Mean loss 2.5976 is not a quality score.

One continuing coding agent scored all 72 neutral responses across three shards and locked their hashes before unblinding; other rubric dimensions were null when unassessed. Revised versus original50: 8 score improvements, 0 degradations, 16 ties; critical cases 5/24 versus 7/24 (base 5/24). Calibration mean 1.167 to 2.500 and detail 0.667 to 1.667; chronology and correction unchanged at 1.333. All three misleading premises remain critical and all six revised explanations score below3/4. Revised/base paired score counts: 7 improve,4 degrade,13 tie. Mean tokens base/original/revised407.04/137.17/80.92; truncations5/5/1.

Retain revised data as a diagnostic, do not promote the adapter or extend epochs. Next proposed change is lr1e-5 at the same revised data/order/50 updates with everything else fixed; no run authorized. Fresh source-separated cases are required; v2 is now development evidence. Single reviewer, small/control-imbalanced sample and target exposure4407 versus2404 limit inference; no causal LR result or historian validation. Full review, paired case table/CSV and all-response audit are preserved in `evaluation/runs/kaggle_data_revision_v2_v1/`. At user request the remote wait used quiet10-minute scheduled checks; final checks passed and the checker was deleted.

## 25 October 2026 — controlled 1e-5 experiment prepared; upload blocked

User authorized fresh adapter, exact same revised 50 rows/order/50 actual updates and all other settings, with lower LR as a hypothesis. Added immutable v3 holdout: 24 cases/six source-separated works, six each chronology/premise/evidence calibration/detail, with positive and conditional controls. All 24 inspected v2 cases are regressions, separately reported. Exact byte data/lock/order and 4,407 supervised tokens preserved. Frozen private Kaggle notebook compares base/saved revised1e-4/new revised1e-5 and checks same initial adapter bytes. Clean CPU notebook unpack/compile and tokenizer-only mask/order/budget checks passed. Additional fresh-reference/prompt versus original500/revised526 message 16-word check found no matches. 244 CPU tests pass; two historical fixture assertions were updated to handle additive holdout inventories and an existing excluded-generated-passage rejection while preserving historical snapshots. No frozen old runtime helper/data/adapter/result was rewritten.

Kaggle read-only status confirmed the saved comparator complete. Submission was blocked twice by automatic approval review: direct authorization for the specific third-party code/data/prompt export is required, and file-based prior approval/public-content audit did not satisfy it. `training/lr_1e5/upload_review.json` documents the exact 14-file allowlist, public historical source provenance, no credentials/private user documents/reference answers. Pending direct user approval to upload that notebook, the same50 chats and48 evaluation message sets to private Kaggle and run. No optimizer update or response was generated; no local model weights/GPU were loaded. Both adapters stay experimental; no new full-parent epoch.

## 25 October2026 — controlled1e-5 test complete and reviewed

Direct user “yes” resolved the earlier private-Kaggle export-review block. Private `thanvirdiouf/trotsky-revised-lr1e5-50-updates/1` completed50 actual finite updates (128.29sec update loop) and144 responses. Same50 revised rows/order/4407 supervised tokens/initial adapter bytes, only LR1e-4→1e-5;14 runtime checks passed. Preserved87,201,766-byte ZIP,41 entries, CRC/path/byte/frozen-input/checkpoint/order/counter/probe/journal checks passed. A local relative-path formatting error after successful unpack/validation was fixed; strict verification was rerun without re-extraction or model work. No local model weights/GPU used. Historical data/helpers/adapters/results unchanged.

One continuing agent explicitly scored144 neutral answers over3 shards, with exact response hashes/evidence notes, locked before unblinding. Fresh v3:5 improved/10 degraded/9 tied; chronology1.333→2.167, correction1.500→1.333, calibration3.000→2.000, detail2.000→1.333; critical2→3. Positive pre-cutoff chronology stays1.25; positive-control≥3 count4/14→3/14, detailed answers≥3 count2/6→1/6. Lower LR therefore does not show overall knowledge retention with calibration retained.

Inspected v2 separately:5 improved/4 degraded/15 tied; critical5→4; all3 false premises remain critical and0/6 explanations reach3. Exact24/24 old base/control prompt/decoding/input/output-token/text reproduction passed; repeat main-score agreement13/24 and19/24 (all changed scores differ by1), critical agreement23/24 and24/24. These are reviewer differences, not model changes. Scores retained, not silently reconciled. Both adapters experimental; no next run/added epoch/promotion. V3 now development evidence. See `evaluation/runs/kaggle_lr1e5_v1/` quality review, audit, paired CSV, control/retention diagnostics, reproducibility and verification.253 CPU tests passed with9 targeted postprocessor checks after regression-reference verification.

## 25 October 2026 — continuation audit

Verified that handoff, LR continuation/status, machine-readable decision and result index cover the completed private run, frozen inputs/identity/hashes, local adapter versus remote optimizer checkpoints, 144 locked judgments, reproducibility/scoring limitations, checks and experimental/no-epoch decision. Added the latest recommendation explicitly: independently re-review the 24 inspected v3 cases first with no GPU work; only then consider a new-primary-case four-condition base/1e-4-adapter × no-passages/same-curated-passages test. Neither is started; no further training selected. New reviews must use separate artifacts and preserve original scores. No continuation monitor or automatic rerun was created for the completed run.

## 25 October 2026 — frozen source-grounding comparison completed

- New24-case/six-work freeze with original/revised training/prior-evaluation exclusion plus one near-duplicate family. Exact passage/prompt/rubric hashes fixed before outputs; source audit preserved with inherited-context correction.
- Private Kaggle `thanvirdiouf/trotsky-source-grounding-comparison/1`:96 answers, untouched base and saved revised1e-4 each with/without identical passages. GPU0; zero optimizer updates; exact saved adapter tensors and matched input tokens verified. Both saved adapters still match original archives.
- Local verifier initially compared newline-bearing checkpoint identity to newline-free run identity; corrected before answer inspection, regression-tested, original extraction/correction retained. No rerun or scoring-rule change.
- Separate clean-context coding-agent reviewer read96 neutral answers, provided explicit source-backed judgments and locked hash `d646f58fa659924f9487625b43e6a733e65941f90e7a7c88c4df459b99a86053` before unblinding. Not historian validation; style can expose conditions.
- Passage effects: base19 improved/0 degraded/5 tied; adapter22/1/1. Critical cases13→3 and9→1; negative premise corrections1→3 of4 for both; supported controls score≥3 become12/14 for each. Grounded adapter/base7/6/11, critical1 versus3, detailed answers≥3 at5/6 versus6/6.
- Main agent inspected all eight sub3 grounded answers after lock. Shared Rakovsky certainty failure survives explicit evidence; grounded base adds Eastman/Blumkin replacement stories and source-role/date errors persist. Decision: grounding first, grounded base simpler reference; adapter remains challenger for possible safety advantage. No training/epoch/LR sweep/promotion. Eight distinct targeted CPU tests pass.
- Evidence: `evaluation/runs/kaggle_grounding_v1/` and `evaluation/fresh_holdout_grounding_v1/`; decision mirrored in `training/next_experiment_decision.json`.
