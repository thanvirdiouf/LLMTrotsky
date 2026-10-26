# Trotsky’s ghost — present-day chatbot

The active goal is a **clearly fictional Leon Trotsky ghost living in the present
day**, with no 1940 knowledge cutoff. The chatbot distinguishes historical
record, dated contemporary evidence and newly imagined interpretations. Its
active profile is selected in `config/active_chat.json`; archived historical
experiments and both adapters remain preserved. **Training stays paused.**
[Active ghost guide](docs/PRESENT_DAY_GHOST.md).

These build commands require the complete local research archive. A fresh
clone contains the code and persona configuration; restore the reviewed source
inputs as described in [the distribution guide](docs/GITHUB.md).

```bash
python3 scripts/build_ghost_chat.py
python3 scripts/ghost_chat.py --dry-run --question "How would you assess renting an A40 on Runpod today?"
```

Forty-three focused ghost/source/harness/comparison tests pass. The fresh
matched-source comparison separates two failures: direct replies engage with
the conversation but invent certainty, while checked replies preserve reviewed
facts but miss task changes or reject formatting. The next regression classifies
the current task in the existing selection call and tests those application
fixes. The active collection has ten packets/39 propositions. Training remains
paused; neither pipeline is promoted as reliably grounded.
[Measured progress](docs/PRESENT_DAY_GHOST.md).

The following describes the preserved historical research archive.

**Historical V2 was implemented, but its Kaggle comparison ran out of memory
after 14 direct-control answers, before the checked arm.** It adds
persistent source state, reviewed claim records, exact quotation/citation
rendering and a select/draft/check pipeline. All 21 grounded-chat tests pass;
the offline source-recall regression is 21/21. Semantic quality is still under
evaluation. V1 artifacts and both saved adapters are preserved; training stays
paused. [V2 guide and comparison](docs/GROUNDED_CHATBOT_V2.md).

The prototype connects the **untouched base model to bounded
retrieval**, preserving source context, dates, speakers and uncertainty. Seven
source packets, eight new single questions with oracle diagnostics, and four
sustained conversations completed **32 verified inference-only responses** on
private Kaggle version 1. A same-agent, unblinded review finds interpretation
and conversation failures despite relevant evidence, alongside one follow-up
retrieval miss. The strict review flags 14 critical answers, including three
quotation-fidelity cases; 11 have other substantive failures.
Both adapters remain experimental and further training remains paused.
[Prototype and commands](docs/GROUNDED_CHATBOT.md),
[results and limitations](evaluation/runs/kaggle_grounded_chat_v1/QUALITY_REVIEW.md).

Source-grounded historical persona experiments: corpus preparation, reviewed
training data workflows, QLoRA training tools, and controlled comparisons of
prompting, adapters, and supplied source passages.

**License:** original software and documentation are available under
[GPLv3 only](LICENSE). Historical texts, translations, datasets, and model
artifacts have separate terms; see [NOTICE.md](NOTICE.md).

**GitHub distribution:** this repository contains the source code and project
documentation. The corpus, curated data and review locks, generated notebooks,
and model/run artifacts remain local and are excluded from Git. The status
notes and artifact links below describe that local research archive.
See [setup and publication instructions](docs/GITHUB.md) for what works from a
fresh clone and how to publish the repository.

## Source-grounding comparison complete — 25 October 2026

All **96 answers on24 fresh cases** completed on Kaggle without retraining. A separate clean-context coding agent scored randomized answers and locked scores before unblinding. Passages improve **19/24 base** and **22/24 saved1e-4** cases; critical failures fall **13→3** and **9→1**. Both grounded models retain12/14 positive controls at score≥3 and correct3/4 negative premises. Grounded adapter versus grounded base:7 improved/6 degraded/11 tied, with1 versus3 critical cases. **Prioritize grounding; use grounded base as the simpler reference, retaining the adapter as an experimental challenger.** Both still accept the Rakovsky proof premise despite explicit uncertainty; inspect this before any targeted training. Both adapters remain unchanged and experimental; no extra epoch, LR sweep or promotion. [Results](evaluation/runs/kaggle_grounding_v1/QUALITY_REVIEW.md), [decision](evaluation/runs/kaggle_grounding_v1/DECISION.md), [failure inspection](evaluation/runs/kaggle_grounding_v1/FAILURE_INSPECTION.md). These24 are now development evidence; six source clusters and one coding-agent scorer limit generalization.


The archived experiments prepared a source-grounded late-life Leon Trotsky persona for Qwen3.5-9B. Their historical date is **19 August 1940**, retained in `config/persona.json` for reproducibility. The active application now uses `config/persona_ghost.json` without that knowledge cutoff. GPU-intensive work runs elsewhere.

Start with [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) for current status and decisions, [the experiment plan](docs/TRAINING_PLAN.md) for LoRA/RAG choices, and [the readable seed examples](datasets/seed_preview.md) for concrete conversational data.

## Evaluation re-review — 25 October 2026

All72 saved answers on24 fresh v3 cases have a separately locked source-based second review. The 1e-5 versus1e-4 tally changes to **8 improved / 8 degraded / 8 tied**; temporal rejection improves but pre-cutoff chronology stays tied, calibration remains worse, and critical counts remain6/2/3 for base/1e-4/1e-5. This coding-agent reassessment had seen the old summary and is not independent historian validation. **Training is paused; both adapters remain experimental.** [Review, disagreements and verification](evaluation/runs/kaggle_lr1e5_v1/independent_review_v3/REVIEW.md). Next: [fresh four-condition source-grounding comparison](evaluation/SOURCE_GROUNDING_NEXT.md), with no retraining; that comparison is now complete as linked above.

## Historical completed controlled lower-learning-rate test — 25 October 2026

The fresh1e-5 adapter completed **50 verified updates and144 matched responses**, with the same revised50 examples/order/4407 supervised tokens and matched initialization. All14 runtime checks and local artifact checks passed. On24 fresh cases,1e-5 versus1e-4 gives **5 improved,10 degraded,9 tied**; chronology improves but calibration/detail decline. Critical cases are base6/24,1e-4 2/24,1e-5 3/24. Inspected v2 regressions are reported separately. One continuing coding agent locked144 neutral scores before unblinding; repeat scoring varies. **Both adapters remain experimental; no added epoch or next training is selected.** [Full review](evaluation/runs/kaggle_lr1e5_v1/QUALITY_REVIEW.md), [verification](evaluation/runs/kaggle_lr1e5_v1/verification.json), [next decision](training/NEXT_EXPERIMENT.md).

The preceding revised-data experiment remains preserved in [its original review](evaluation/runs/kaggle_data_revision_v2_v1/QUALITY_REVIEW.md):8 improvements/0 degradations versus original-data step50, with unresolved failures. The lower-rate test does not turn that diagnostic gain into a reliable deployed adapter.

## Current artifacts

- `trotsky_corpus.txt`: original supplied corpus, preserved unchanged.
- `corpus_mia/raw_html/`: a fresh cache of **1,913 linked MIA HTML pages**, including nine explicitly selected works hosted elsewhere on marxists.org. This is the available linked archive within the configured scope, not proof of complete lifetime works.
- `corpus_mia/documents.jsonl`: paragraph-level extraction with provenance and separate metadata/editorial/note roles.
- `corpus_mia/cleaned_candidates.jsonl`: **1,823 candidate text pages**, with conservative, logged repairs. The bulk corpus remains unreviewed and must not be treated as blanket training approval.
- `corpus_mia/qa_candidates.jsonl`: **1,606 unreviewed dialogue candidates**, many from the Dewey hearings.
- `corpus_mia/catalog.jsonl`, `source_families.jsonl`, `duplicate_edges.jsonl`, and `review_queue.jsonl`: source grouping and curation aids. Automatic grouping is not exhaustive.
- `datasets/train_seed.jsonl`: **500 coding-agent-reviewed seed examples across 123 source documents**, including 65 multi-turn conversations; exact evidence and review history are retained separately. This reviewed set is sufficient for the initial small pilot; it is not independent historian validation.
- `evaluation/cases.jsonl`: **36 initial evaluation cases** with rubrics and source evidence. Whole grouped works are reserved from seed training. Serial and dual-GPU 9B runs completed all 36 cases with identical answer text, including 13 truncations. A [provisional coding-agent rubric review](evaluation/runs/kaggle_361155394/QUALITY_REVIEW.md) flags critical failures in nine cases; independent scoring remains outstanding.
- `notebooks/kaggle_qlora_smoke.ipynb`: standalone two-step Kaggle QLoRA preflight, with the 500 reviewed examples, explicit token-mask inspection, text-only adapter/frozen-base checks, memory/gradient reports and fresh-base adapter reload/generation. [All 500 real-tokenizer checks passed](training/token_mask_preflight.json); The [two-step GPU smoke test passed](https://www.kaggle.com/code/thanvirdiouf/trotsky-qlora-training-smoke-test?scriptVersionId=361399211) with 8.48 GiB peak allocated memory and matching generation after adapter reload. See [training instructions](training/README.md).
- `notebooks/kaggle_prompt_baseline.ipynb`: standalone Kaggle benchmark with one model copy per GPU, concurrent workers, resumable journals and timing. Live dual-GPU execution completed 36 cases in 12 minutes 28 seconds, plus 95 seconds of checkpoint preparation. See [the verified run](evaluation/runs/kaggle_361155394/README.md) and [the remote workflow](docs/REMOTE_MODEL_WORKFLOW.md).
- `notebooks/kaggle_source_validation.ipynb`: original six-pair provided-source condition, completed on Kaggle with ten truncations. A [provisional paired review](evaluation/runs/kaggle_361239681/QUALITY_REVIEW.md) finds source-specific gains and one critical failure per arm.
- `notebooks/kaggle_source_validation_concise.ipynb`: separate follow-up applying the same 120-word instruction to both arms, with word-count diagnostics and original model/decoding settings. See [the concise protocol](evaluation/source_validation_concise/README.md).
- `prompts/generate_examples.md`: a source-grounded generation/acceptance protocol for expanding the data.
- `scripts/remote_model.py`: a bounded remote generation/evaluation client; see [the remote workflow](docs/REMOTE_MODEL_WORKFLOW.md). Mock tests, real-data request previews and rejection of an unreviewed source passage passed; no model service has been called.
- `QUALITY_REPORT.md`: verified inventory, integrity checks, and remaining curation limits.

Composition dates, publication dates, URL years, and web-update dates are distinct. Candidate documents can contain other authors, quoted opponents, OCR defects, or unlabeled commentary. The pipeline preserves evidence and requires individual curation instead of silently treating every archive page as Trotsky's own speech.

## Reproduce with the complete local research archive

Python with `requirements.txt` dependencies is sufficient for preparation. No model weights or GPU are needed.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/extract_mia.py --input corpus_mia
python3 scripts/build_catalog.py --input corpus_mia
python3 scripts/build_seed_dataset.py
python3 scripts/build_evaluation.py
python3 scripts/clean_candidates.py
python3 scripts/validate_project.py
python3 scripts/update_handoff.py --note "Describe the verified milestone"
```

The evaluation builder requires a finished crawl and a current catalog, and rejects source-family overlap with the seed. The seed builder rejects silent changes to locked review evidence. See the dataset README before revising a review.

Alternatively, `python3 scripts/prepare_corpus.py` runs the six offline preparation/validation stages in order and updates the handoff after each successful stage.

To add or resume downloads, only when no other downloader is active:

```bash
python3 -u scripts/fetch_mia.py --output corpus_mia --supplemental sources/mia_supplemental_urls.json
```

This uses a single connection, saved robots policy and delay, redirect checks, an exclusive process lock, and a resumable queue. Public network access may require the execution environment's normal network approval mechanism. Do not run concurrent writers against the same corpus directory.

## Historical source-validation and smoke-test stage

This paragraph preserves the earlier planning snapshot; the 500-example pilot and subsequent audit/comparison have since completed, and the current v2 experiment above supersedes its pending statements.

The original paired review and concise follow-up are complete. The [verified concise run](evaluation/runs/kaggle_361242522/README.md) finished all six pairs without truncation; its 112–141-word answers retain source benefits but still contain conceptual errors and unsupported self-revisions. This does not evaluate retrieval or broad historical reliability. The next concrete step is the standalone QLoRA smoke-test notebook, followed by an initial pilot using the existing 500 examples. The two-step compatibility smoke test passed; the full 500-example pilot has not run. The inspected six questions are development evidence; fresh evaluation across other works and failure types is required before broad conclusions or scaling.
