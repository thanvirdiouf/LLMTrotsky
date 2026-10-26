# Fresh source and conversation tasks, version2

A separate AI reviewer authored two new independent source questions and an
eight-turn conversation after saving its previous source/conversation judgments.
It read only the full source collection, without the claim catalogue, current
code, source-curation proposals, checker calibration or V24 outputs. Questions
and reviewer checks were saved before corresponding base-model inference and
remain unchanged in `config/ghost_fresh_transfer_evaluation_v2.json`.

The independent tasks distinguish report routes/hypotheses/unknown evidence
and an ILO percentage denominator plus conditional local applicability. The
conversation concerns independent reading groups and a joint forum. It tests
substantive disagreement, a change in decision rights, two source-only pauses,
a delayed return and a 40–55-word closing. It allows different political choices.
Each call gets at most two controlled packets; the conversation uses the same
Weisbord/New Leader pool. This is not natural-retrieval recall measurement.

The author selected facts from full passages, not facts chosen to fit the
catalogue. A missing catalogue entry remains a representation limitation;
unknown final fate, informant identity or local-office characteristics remain
genuine evidence limits. Support and usefulness are scored separately. No
pretraining novelty, human historical gold or representative reliability claim.

## Pre-inference application correction

The continuing agent inspected the new design before inference. A deterministic
preflight found that the earlier memory heuristic captured the opening scenario
but missed the natural directive Revise the forum: it only recognized revising
the hypothetical. The unchanged opening/revision texts and source questions
were not rewritten to suit that implementation.

The retention rule now recognizes sentence-initial revise/update/modify/amend
when an original user scenario is already stored. It is independent of the
scenario noun. Ordinary revision instructions without existing scenario context
do not start this memory. Original messages remain unverified, whole and bounded;
no summary or verified source claim is created. The heuristic still does not
understand every possible revision or retain every objection.

Before/after local traces show delayed context changing from only turn1 to
turns1/4. A new test covers different nouns/verbs, provenance and the ordinary
editing boundary. This is a preflight-driven fix after the programmer saw the
questions, so the run is a fresh-task development diagnostic, not untouched
held-out application validation. Expectations remain hidden from the tested
model, and no test answer is used for training.

V25 freezes the same profile, source packets and42-claim ledger as V24. Only
the helper, new model questions and protocol hash differ. The builder now also
rejects supplemental canonical statements that cannot pass the actual factual
renderer; this validation adds no GPU inference or new runtime payload. All77
focused tests pass; the full350 suite retains the same two historical README
snapshot errors. The four supplement-boundary tests also pass after the final
renderability guard.

CPU-only synthetic construction passes ten selectors, maximum3,257 tokens,
with original/revised user context present at the delayed return. This is not a
bound on real future reply lengths or later draft/check inputs. The runtime
retains its5,120-token pre-inference cap. Full source blocks and all author/date/
speaker/uncertainty metadata remain supplied; expectations/rubrics/provenance
are omitted from the model cases.

V25 is bounded to ten replies, at most30 calls plus one short memory probe on
the existing private free-Kaggle notebook. No training, adapters or paid compute.
After exact download, token reconstruction and frozen replay, export a neutral
source/response packet for review in actual displayed histories. Keep pipeline
statuses and draft/checker outputs out of that semantic review.

## Executed V25 result

V25/saved27 passed frozen replay and token reconstruction: ten replies/28 calls,
no budget rejection or truncation,11.32GiB. Both original and revised forum
messages reach all delayed-turn stages as unverified context. The model uses
unanimous declaration in its delayed reply, but omits several other powers.
The closing is82 words against40–55.

Masked E review, with all ten exact reply hashes verified, finds a fully covered
opening participation answer alongside major arithmetic/denominator, close-
reading, revised-powers and shortening gaps. Support and usefulness remain
separate. The arithmetic draft produces10.4%, but the application filters it
for lacking a historical basis; its denominator explanation is also incomplete.
A revised-forum draft is filtered by the broad I once guard despite citing a
documented argument. Those causal observations come from a separate implementation
audit, not the displayed-only reviewer.

The next [matched factual comparison](GHOST_SOURCE_READING_COMPARISON.md) tests
whether a minimal source-reading pipeline can improve completeness without
losing source fidelity. It remains experimental; training stays paused.
