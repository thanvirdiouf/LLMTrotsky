# Initial evaluation protocol

## Source-grounding comparison complete — 25 October 2026

All **96 answers on24 fresh cases** completed on Kaggle without retraining. A separate clean-context coding agent scored randomized answers and locked scores before unblinding. Passages improve **19/24 base** and **22/24 saved1e-4** cases; critical failures fall **13→3** and **9→1**. Both grounded models retain12/14 positive controls at score≥3 and correct3/4 negative premises. Grounded adapter versus grounded base:7 improved/6 degraded/11 tied, with1 versus3 critical cases. **Prioritize grounding; use grounded base as the simpler reference, retaining the adapter as an experimental challenger.** Both still accept the Rakovsky proof premise despite explicit uncertainty; inspect this before any targeted training. Both adapters remain unchanged and experimental; no extra epoch, LR sweep or promotion. [Results](../evaluation/runs/kaggle_grounding_v1/QUALITY_REVIEW.md), [decision](../evaluation/runs/kaggle_grounding_v1/DECISION.md), [failure inspection](../evaluation/runs/kaggle_grounding_v1/FAILURE_INSPECTION.md). These24 are now development evidence; six source clusters and one coding-agent scorer limit generalization.


`cases.jsonl` contains **36 initial cases**: six source-backed validation cases, six source-backed test cases, and 24 behavioral checks. In the prompt-only baseline, reference passages and checklists are **outside** the model input. The source family manifest reserves entire grouped works, not just question rows. Serial and dual-GPU baseline runs have completed; [the provisional agent review](runs/kaggle_361155394/QUALITY_REVIEW.md) covers all 36 answers. Independent human scoring remains outstanding.

Run the same prompts against a baseline persona prompt, LoRA, and—when implemented—retrieval variants. Use the same historical date, model checkpoint, generation settings, and chat-template/thinking policy. Blind the model labels when reviewing responses.

Score each applicable dimension from 0 to 4: 0 = clear failure, 1 = major problems, 2 = mixed, 3 = good with minor issues, 4 = fully supported and appropriate. Leave an inapplicable dimension blank instead of manufacturing a score.

- **Temporal consistency:** respects 19 August 1940, distinguishes past predictions and user-supplied hypotheticals from known outcomes, and can still discuss events already known in 1940.
- **Source faithfulness:** answers the question accurately within the evidence; preserves conditions and uncertainty.
- **Attribution:** separates Trotsky, opponents, translators, editors, and later commentary.
- **Persona voice:** coherent first-person historical viewpoint without a forced speech or generic encyclopedia narration.
- **Conversation:** follows the question, requested length, and preceding turns.
- **Calibration:** admits limits without inventing memories, quotes, citations, or false precision.

Flag critical failures separately: fabricated direct quotations/citations, invented intimate historical memories, definite post-cutoff hindsight presented as memory, or systematic speaker/editor misattribution. Do not let a high average style score hide them.

The checklists describe what an answer must respect, not one mandatory wording. A conditional response to a later event supplied as a hypothetical can be valid. A confident description of its real later outcome as independently known is not. Historical fidelity is not agreement with Trotsky's claims, and source passages can contain mistakes or partisan judgments.

Inspect failure categories and concrete examples before changing the training recipe. An LLM judge can assist later, but should be calibrated against independent human ratings rather than treated as an infallible historical evaluator.

Base-model pretraining may already contain these books. This is a holdout from this project's supervised training, not proof of novel historical knowledge. RAG evaluations may retrieve the held-out texts; label that condition explicitly.

The next [provided-source validation experiment](source_validation/README.md) uses only the six validation questions, each with and without its verified source passage. Its source arm deliberately supplies reference evidence; checklists remain withheld. This measures an oracle evidence condition, not retrieval performance. The original 36-case file and prompt-only baseline stay frozen. All six validation questions come from one chapter of one work, so the result cannot establish general historical reliability.
