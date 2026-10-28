# Offline review after saved46

The saved46 stop rule applies: no more selector-prompt retries on the familiar
five cases. No Kaggle job is running or prepared. Active9B/v3, adapters, training,
frozen questions/expectations and earlier artifacts remain unchanged.

The implementation review is unblinded development review, not independent or
human validation. The observed failures belong to different layers:

| Layer | Evidence and implication |
| --- | --- |
| Source representation | The old Eastman summary compresses seasonal timing, drafting actor and group decision. Selection cannot recover wording absent from a record. |
| Expectation scope | The Blumkin rubric requires broader reform context. The source separately reports conversation advice and a letter carried through him to friends; a missing broader-context record does not prove the selected direct advice is false. Frozen labels remain unchanged. |
| Target/control interpretation | Saved46 still answers the original hearing question for a scope-only target. One renamed data key improves only the ILO control; intent routing remains unreliable. |
| Source selection | Preserving selected statements prevents model factual rewriting. It establishes neither relevance nor complete answers. |
| Retrieval | These trials supply controlled packets and all available summaries. They do not measure live retrieval quality. |
| Presentation/conversation | Source panels have not established readable answers, sustained dialogue or a consistent present-day fictional voice. |

## Application-owned request boundary

`scripts/grounded_request_boundary.py` is an offline, unintegrated contract for
explicit application events. An `ask` event requires its own current question;
prior turns remain separate opaque context. Structured source-scope and detail
controls update immutable state and produce no question request. Unknown fields,
unconfigured sources, duplicate scopes, missing questions and system-role prior
turns are rejected. Compiling a request records no answer or trusted factual
memory. This module contains no model, prompt, classifier or inference adapter.

The calling application supplies event intent. This contract does not authenticate
the caller or classify free-form language. In particular, text saying “use only
the excerpt” inside an explicit `ask` event still produces a question request;
there is no keyword-based runtime answer mapping. Natural-language follow-ups
and scope changes require a separately evaluated product policy. A UI source
filter event is a different input from a textual conversational instruction.
No integration or runtime activation is implied by these unit tests.

Source scope is a nonempty subset of the configured bounded collection, not a
preselected one/two-packet answer. Retrieval can later choose packets within that
scope; its packing limits and relevance evaluation remain a separate stage.

Six new tests cover control events producing no question, explicit current
question versus prior context, immutable state, opaque text without hidden intent
routing, source/field validation and role boundaries. Next work may exercise an
offline request-to-retrieval harness on synthetic fixtures, with no model or
changes to the active route. Only a justified new architecture design can precede
future inference; familiar selector-prompt retries remain stopped.

## Source precision extension

`config/ghost_reviewed_precision_plan_v1.json` pins all parent51-record ledger,
profile, collection and manifest bytes. `scripts/extend_reviewed_source_ledger.py`
creates a new opt-in53-record snapshot at `evaluation/ghost_reviewed_coverage_v3/`,
preserving all51 original claim objects and the exact profile/source bytes.
The two additions are source paraphrases:

- Eastman: the author's autumn1925 account explicitly identifies the Political
  Bureau majority, the Opposition group decision and the sign-or-struggle
  ultimatum, retaining the retrospective account status and1928letter date.
- Blumkin: the informative letter's carrier, Moscow recipients and reform
  position are explicitly attributed to the letter, separately from conversation
  advice. Letter wording is not labelled as a verbatim spoken reply to Blumkin.

Exact full source-block anchors, speakers, dates and context are preserved in
the ignored supplement and new source review. The earlier historical-only guard,
mixed-domain supplements, frozen51-record experiments and evaluation questions
are unchanged. The new snapshot is unused by any runtime or inference job.
Three tests reject changed parent/supplement bytes, addition-count changes and
attempts to overwrite an original claim, and check source/profile preservation.

This extension repairs representation precision only. Machine anchor validation
does not prove semantic truth; source curation remains the coding agent's own
review. It does not fix task selection, validate the rubric, justify training or
permit promotion. Fixed fresh transfer questions remain unrun.

Validation completed: all158 focused ghost tests plus six boundary tests pass.
The53-record snapshot's fingerprint is
`1ac4132bb0aff970c4a7f859106fffe7fb3308ce5635627ad8a3e10fe221e6e4`.
Ignored source review and offline identity checks record the unchanged original
51objects/profile/collection bytes. Zero model calls; no active integration.
