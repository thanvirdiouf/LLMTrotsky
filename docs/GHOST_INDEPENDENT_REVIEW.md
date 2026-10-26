# Second review and result-driven continuation

The user explicitly requested a second source-based review, followed by repeated
result-dependent work. A fresh-context reviewer was delegated the V14 worksheet
without parent history, prior case labels, model verdicts or earlier reports.
It saved judgments before authoring the next questions. This is a separate
fresh-context AI instance from the coding agent’s model family, distinct from
the Qwen base under test. It is not human historian validation; correlated
blind spots with the coding agent remain possible.

The review covers all ten displayed replies and five candidates separately.
Every displayed-text hash matches the verified journal. Its input worksheet
hash and declared files read are recorded in the local review metadata. Source
identities and actual conversation context remain visible; the procedure is
blinded to prior scores and model approvals, not to the questions or sources.

The second reviewer confirms the unnecessary prediction-scope refusal, the
unsupported universal about previous technological revolutions, repetitive
argumentation and final displayed-length failure. It distinguishes the useful
source-only and insufficient-evidence answers from those failures. It treats
some absolute political rhetoric as ambiguous conditional/philosophical claims,
rather than declaring every strong view factually false. The continuing agent
accepts that distinction: a permissible preference, an unsupported historical
assertion and an ambiguous conjecture require different labels. No uniquely
authentic modern Trotsky answer is claimed.

Local originals are under `evaluation/runs/ghost_smoke_v14/second_review/`:
`independent_review.json`, `INDEPENDENT_REVIEW.md` and `fresh_questions.json`.
They are research artifacts excluded from Git. The reviewer-authored public
case design is `config/ghost_second_review_evaluation_v1.json`; exact questions
are preserved while the execution wrapper makes source context explicit.

## Next controlled iteration, preregistered before outputs

V15 runs the existing V14 application on ten reviewer-authored replies: four
single source-limit questions and six sustained cooperative discussion turns.
Reviewer-selected packet context is fixed per single case and throughout the
conversation. This isolates interpretation and answer usefulness, not natural
retrieval recall. All actual displayed replies are stored; the answering model
keeps the application's last-four-message context and omission count. Expected
checks, ratings and review rationales are excluded from model cases.

Before inspecting V15 outputs, the proposed V16 intervention is narrow: a mixed
source answer should retain a reviewed historical factual component when no
modern factual component applies. Prefer the answer planner's original claim
order over application-added historical basis. A rejected interpretation must
not erase a separately supported canonical fact. Exact reviewed statements,
dates, attribution and uncertainty remain unchanged; no rejected prose is
salvaged or silently edited into evidence.

V15 baseline is frozen before this intervention. V16 will retain its profile,
collection, ledger, questions, context policy, evaluator and base generation
settings. Separate runs are baseline-first, not randomized. Later histories can
differ because actual replies differ. Compare source-supported usefulness,
unsupported additions, unnecessary refusals and conversation quality separately;
model approvals are not the outcome score. Do not claim a broad improvement
from the old inspected question alone.

Each private Kaggle run stays within ten replies/thirty evaluation calls plus
one separate memory probe, no training/adapters/paid compute. Both original
adapters remain experimental and unchanged. Milestones are committed/pushed.
After review, choose the next intervention from the new evidence rather than
changing these questions until they pass.

## Repeatable review exports

`python3 scripts/export_ghost_review.py --version ghost_smoke_vN` exports an
integrity-checked `blind_review/review_packet.json` from the verified answer
journal and frozen source bundle. It includes exact source context and actual
conversation order, excluding pipeline status, model verdicts, prior scores and
rejected candidates. A separate private mapping links neutral review IDs to
run keys. Review displayed replies first; inspect candidates later for diagnosis.
Calibration judge outputs are refused as chatbot-review inputs. This differs
from the existing full diagnostic worksheet, which also exposes candidates.

V15 is submitted as private saved version 17. V16 is frozen after the narrow
historical-component change. All payloads except `ghost_chat.py` and derived
protocol hashes are byte-identical, including reviewer-authored model questions,
profile, collection, ledger, evaluator and memory probe. Local regressions prove
that an invented quoted opinion cannot erase the planner-prioritized canonical
prediction; view-only replies remain separate and mixed source behavior is not
keyword-gated. Fifty-four focused tests pass. Full suite: 327 tests, the same
two pre-existing historical README snapshot errors; locks remain preserved.


V15/saved17 and V16/saved18 are now complete and verified: ten replies/22 calls
each, no input/output budget failures, about 11.32 GiB peak. All ten displayed
replies are byte-identical; the historical mixed-fact fallback does not trigger
on this new set. No fresh quality gain is claimed. A second reviewer receives
neutral randomized packet labels without prior statuses or model verdicts.
Both runs expose two invalid singular task labels. V17 separately tests finite
selection decoding; semantic usefulness remains a separate outcome. See
[constraint experiment](GHOST_SELECTION_CONSTRAINTS.md).
