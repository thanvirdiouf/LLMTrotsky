# Standalone factual-source audit diagnostic

V29/saved31 is frozen/token verified: ten replies,18 evaluation calls, no input
rejection or truncation,11.32GiB peak. All five reading controls are identical
to V28. The block-reference map reaches composition in three cases and displays
two answers, but both displayed answers contain unsupported details approved by
the checker. One adds an unsupported first name and former Opposition membership,
and asserts a reconstructed deportation as settled. The numerical answer gives
10.4% with the female-employment denominator, but invents a user-implied10.2%
figure. Other cases block on a nonliteral quotation, malformed JSON or an
overlong field. No mapped route is promoted and fresh transfer remains unrun.

The next bottleneck to isolate is factual checking. Further mapping changes
would not fix the demonstrated approval of unsupported names, memberships,
reported roles or hypothetical consequences. `source_audit.py` defines an opt-in
standalone factual auditor. Its prompt uses full passages and exact candidate
text without a map, canonical cards, fictional speaker identity, conversation
history or prior model verdicts. It asks for clause-level comparison, preserves
reported statements as attributed accounts, and forbids silently repairing the
candidate in the reason. It returns the existing six-Boolean verdict schema.
It is still the same fallible base model, not independent validation.

V30 tests that auditor on five fixed valid/invalid pairs from Rakovsky and
Bessedovsky. Topics are unsupported first name, unsupported Opposition membership,
hypothesis versus independent confirmation, reported information versus personal
witnessing, and an unnamed informant versus the report's subject. The primary
coder authored expectations from the bounded sources after seeing V29 failures;
this is a preregistered development calibration, not human gold or fresh transfer.
Candidate strings, expected decisions, local rationales and hashes are frozen
before inference. Expected labels/rationales are withheld from the base model.

Candidates are diagnostic test data, including deliberately false statements.
Only judgment JSON is a recorded response; candidates never become chatbot
answers or trusted memory. The existing scorer separates false acceptance,
false rejection, malformed/missing judgments and execution blocks. Ten candidates
use at most ten calls plus the usual separate memory probe in the same private
free Kaggle notebook. Model, quantization, greedy generation, sourceplanv4 and
reference date stay pinned. The neutral policy is not installed in either reader,
ordinary conversation or the active v3 pointer. Old audit configurations keep
their original behavior.

Two new tests check exact candidate transmission, absence of labels/maps/cards,
full supplied context, empty trusted memory, replay, missing/truncated/duplicate
JSON verdict rejection and unknown-policy failure. Existing source and engine
checks remain unchanged. After exact-version download/replay/token verification,
score against the frozen expectations and read the reasons for candidate reversal.
Do not count malformed JSON or omission as successful semantic rejection.

If the new auditor rejects the known false details while preserving valid
source-qualified answers, test it as an opt-in reader auditor and then reassess
fresh tasks; calibration success alone cannot justify deployment. If it still
approves these simple false details, do not treat self-audit as a reliable gate
or train on its approvals. Document the failure and prefer inspectable source
presentation or a separately authorized stronger evaluator before trusting free
factual synthesis. Both adapters remain experimental and training stays paused.

All88 focused ghost tests pass. Full361 retains the same two historical README
snapshot errors, with frozen locks unchanged. CPU preflight exercises ten
requests, maximum1,913 input tokens below5,120, with expected metadata withheld
and trusted memory empty. These synthetic approvals are contract fixtures, not
semantic audit results.

V30 was submitted as private Kaggle saved32 after fe4b739 was committed and
pushed. Results are pending; the neutral auditor remains a diagnostic only.

V30/saved32 is now verified:4correct approvals/1correct rejection/1false
approval/1false rejection/3format blocks;0of5pairs both correct. No integration.
See [thinking-mode audit diagnostic](GHOST_THINKING_AUDIT.md) for the next
same-base inference test and [source inspection](GHOST_SOURCE_INSPECTION.md)
for the new model-free evidence view.
