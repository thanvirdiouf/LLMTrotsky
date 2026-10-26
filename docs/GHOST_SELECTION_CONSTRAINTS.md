# Constrained selection as an interface experiment

The reviewer-authored V15 baseline has two blocked replies because the model
emits `premise_correction` outside the facts/view/mixed enum. The earlier plural
synonym recovery does not cover it. Repeatedly accepting new spellings would
not establish a controlled interface. These are schema failures, separate from
source relevance or factual support.

V17 tests a finite token-prefix constraint only for selection. It uses the
native Transformers `prefix_allowed_tokens_fn` mechanism, confirmed against
the pinned local library source and the
[official generation documentation](https://huggingface.co/docs/transformers/v5.18.0/en/internal/generation_utils#transformers.PrefixConstrainedLogitsProcessor).
No new package, model, training step, adapter or inference call is added.

The allowed strings have exactly three fields: facts/view/mixed, brief/normal/
expanded, and zero to four distinct supplied claim IDs. Every subset and every
first-priority claim is allowed; remaining IDs follow source order to bound the
finite space. The current two-packet collection has at most twelve available
claims; larger pools fail explicitly rather than silently losing evidence.
EOS is allowed only after a complete object. Unknown IDs, task spellings,
duplicate claims, extra fields and trailing prose cannot form accepted paths.
The model still chooses among allowed paths from its logits. This constrains
format and membership, not truth, relevance, intent or political judgment.

The six-claim tokenizer preflight builds 1,413 paths with a maximum of 37 output
tokens and verifies sampled exact roundtrips. Runtime caching reuses each
available-ID grammar. Every actual selection logs its policy, whitelist,
candidate count and text-set hash; the frozen verifier checks output membership,
metadata and token-path acceptance using the original constraint code. Other
stages, including opinions and semantic audits, remain unconstrained.

V17 retains V16's profile, collection, ledger, questions, evaluator and ghost
helper. It adds this selection policy to the Kaggle runtime and run identity.
The questions are now inspected regressions, not fresh validation. Primary
outcome: eliminate invalid selection objects without adding calls. Secondary
outcomes must be reviewed: source-appropriate refusal, useful negative answers,
unsupported facts, task correctness and conversation quality. A syntactically
valid irrelevant answer is a failure, even when the model auditor approves it.
The second reviewer will judge neutral packets without pipeline status/verdicts.

This opt-in experiment currently applies to the Kaggle local generation
runtime. The generic remote CLI does not attest or enforce this callback at an
operator's endpoint. No persistent inference service is deployed. The bound
remains ten replies/thirty evaluation calls plus one memory probe on the same
private notebook, with no training/adapters/paid compute. Do not promote the
experiment from structural success alone.


## Verified result and promotion decision

V17 completed as private saved version 19: ten replies/24 calls, zero invalid
selection outputs, no budget rejection or output truncation. Frozen verification
reconstructs every selection path and policy metadata. The structural goal is
met, but the blinded C review finds mixed quality: the historical-summary reply
and ending improve; caption editing and a hypothetical policy request regress
to unnecessary refusals. This is not a clear quality improvement and is not
promoted to the default. Valid syntax does not establish correct task routing.
The next V18 trial returns to unconstrained baseline selection, normalizes the
observed singular/plural factual synonym, and tests reviewed compact forms.
