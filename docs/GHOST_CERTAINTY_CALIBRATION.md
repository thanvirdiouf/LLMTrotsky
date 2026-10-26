# Checker calibration for source certainty

The longer dialogue shows that useful source fragments can coexist with
unsupported confirmation, motive or rights assertions. The next diagnostic
isolates the checker from answer generation: ten independent fixed candidates,
in five valid/invalid pairs. It uses the same untouched base and only the
already reviewed Eastman/Bessedovsky packets. Candidates are untrusted test
inputs, including deliberate falsehoods, never chatbot replies or trusted memory.

A separate AI source reviewer authored the proposal without reading V19–V21
outputs or the other reviewer's judgments. The continuing agent checked the
complete source blocks, dates, actual claim IDs and paired contrasts before
acceptance. Expected labels are AI source-review judgments, not human gold.
The proposal, source snapshots and review notes stay in the ignored local
`evaluation/ghost_certainty_calibration_proposal_v1/` archive. The accepted
questions/candidates, expectations and hash-only source bindings are in
`config/ghost_certainty_calibration_v1.json`; no source-packet snapshots are
published with that file.

The pairs contrast:

- An attributed donation account with independent confirmation of the donation.
- Receiving and donating payment with keeping it for personal use.
- Defending political sincerity while acknowledging theoretical differences with
  asserting complete theoretical agreement.
- A retrospective explanation of pressure with external corroboration.
- Personal acquaintance with some memoir facts and a separately reported purge
  allegation with direct personal knowledge of that particular allegation.

Within each pair, the question, basis IDs and surrounding candidate text match;
only the evidentiary span differs. Dates remain plausible and unchanged. There
is no required uniquely authentic present-day political preference. The third
pair uses Eastman block7 even though the finite ledger has no dedicated claim
for theoretical differences. The full packet must reach the checker; otherwise
it would be a missing-context experiment rather than this calibration.

CPU-only preflight passed all ten actual request constructions. The largest
input is 3,369 tokens under the 5,120 cap. Both packets, including Eastman block7,
are present in every request. Expected decisions, rationales, pair descriptions,
source-binding snapshots and provenance are withheld. Each case resets history;
synthetic preflight verdicts never become semantic scores or trusted facts.
No model weights or inference were used in this preflight.

V22 is frozen for the existing private free-Kaggle notebook: ten checker calls
plus one separate short memory probe, no draft or selection calls, no training,
adapters or paid compute. This fits the existing bounded inference approval.
The unchanged audit evaluator uses strict raw JSON; a block is reported
separately from a semantic rejection and never credited as correct rejection.
The regular ghost helper's new formatting normalization is not silently applied
to this checker-only evaluator.

After exact-version download, frozen replay and token verification, report
false acceptances, false rejections and blocks separately. A pair succeeds only
when both parsed decisions match the reviewed expectations. Inspect rationales
as well as booleans. If false confirmations pass, revise source-status framing
and retest against the frozen baseline. If simple contrasts pass but dialogue
still overclaims, investigate full-pipeline context/generation and coverage
rather than declaring the checker reliable. Neither outcome justifies more
training before the application causes are clearer.

## V22 baseline and V23 framing comparison

V22/saved24 completed and passed exact frozen replay/token verification: ten
checker calls, no input rejection or truncation, peak11.32GiB. All five valid
candidates were approved. Three false variants were rejected, but two were
approved: independent confirmation of donation, and external corroboration of
the retrospective pressure account. Only three of five pairs have both
decisions matching the reviewed expectations. No output blocked.

The rationales expose two distinct problems. One describes the false candidate
as though it had retained the source-account limit, reversing what was actually
written. The other expressly treats a primary account as independent
corroboration. Those are checker reading/concept failures within the supplied
context; they are not retrieval failures.

V23 freezes the same ten candidates/questions, source packets, basis IDs,
expected decisions, model and generation controls. Only the checker system
instruction adds source-status framing: documented statements record what the
author asserts, confident wording/citations are not independent evidence, and
claims of corroboration need an actual independent record or source in the
supplied material. Attributed accounts can still support explicitly conditional
or normative reasoning. No keyword rejection list or changed expected labels.

All ten scored user payloads match V22 exactly; only their system instructions
change. The unscored memory probe also has the new instruction with neutral
padding adjusted to the same5,000-token target. Frozen payload differences are
only ghost_chat.py, memory_probe.json and protocol hashes. The candidate is not
a demonstrated fix before execution.

The reusable reporter binds the exact verified journal, frozen candidate inputs
and preregistered expectation hash:

```bash
python3 scripts/summarize_ghost_calibration.py --version ghost_smoke_v22
```

It reports false approvals, false rejections, blocks and unscored/incomplete
judgments separately. Neither a block nor an empty verdict counts as a correct
rejection. It checks both members before crediting a pair. Three new tests cover
those measurement boundaries and refuse changed expectations or result journals.
All72 focused tests pass; the full345 suite retains the same two historical
README snapshot errors. No historical locks are changed.

V23/saved25 is now complete and token/frozen verified: ten calls, no budget
rejection/truncation, 11.32GiB. It rejects both false corroboration variants and
keeps all five valid approvals, but newly accepts the false theoretical-agreement
variant. Four of five pairs now match both expectations, compared with three
in V22. The false approval rationale paraphrases the candidate as preserving
the very distinction it actually reverses. This is a partial improvement with a
reading regression, not a reliable semantic gate. No general success rate or
human-gold claim follows. Source coverage work remains
[separate and opt-in](GHOST_CLAIM_COVERAGE.md).
