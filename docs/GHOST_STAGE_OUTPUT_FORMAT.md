# Bounded stage output formatting

The V19 transfer baseline completed two outputs that failed strict whole-string
JSON parsing. The compute selector returned one complete selection object and
then an unsolicited planning essay. The source-only audit returned a complete
verdict object followed by one backtick. Neither output was truncated. These
are interface failures, separate from whether the proposed answer is accurate.
The compute essay contained the correct conditional $3.54 calculation as well
as unsupported extra billing assertions; it must never become a displayed answer.

The active ghost helper now has a stage-specific outer-format parser. It first
uses the unchanged strict parser. On failure, only select/check stages may
locate a complete leading JSON object and reparse that exact span with the
original duplicate-field and nonfinite-constant protections. A single trailing
backtick or closing triple backtick can be discarded. Only a selector may also
discard a newline-separated prose postface of at least three words with no
braces, brackets or backticks. Arrays, partial objects, multiple objects,
duplicate fields and nonfinite values still fail. Draft JSON remains strict;
substantive checker commentary remains a failure, including a retraction after
an otherwise approving object. This intentionally leaves some harmless formats
unsupported rather than performing general JSON repair.

A recovered selection remains an untrusted plan: normal field/ID validation,
source-backed fact rendering, interpretation filtering and segment review still
apply. No planner essay supplies a fact, draft, evidence or verdict. Formatting
cannot turn an explicit negative verdict into an approval. Missing verdicts,
duplicate verdict IDs and invalid verdict fields retain their existing handling.
The raw generation stays unchanged in the journal. Each normalization logs its
kind, raw/parsed/discarded fingerprints and discarded character count. As with
other project fingerprints, these hash canonical JSON values, not raw UTF-8
text. There is no extra model call or training update.

Four new tests exercise these boundaries through the actual answer pipeline
and parser, including unsupported assertions in a discarded planner essay,
negative audits, duplicate fields, nonfinite values, multiple objects and literal
braces within JSON strings. All 69 focused ghost tests pass. The full suite ran
342 tests with the same two pre-existing historical README snapshot errors;
old frozen helpers and locks are unchanged. Test logs are local under
`evaluation/runs/ghost_smoke_v20/format_{focused,full}_tests.log`.

This change is prepared separately from the frozen V20 user-context trial.
A later bounded comparison should keep V20's questions, evidence, model and
context controls, change only this helper and protocol hash, then verify the
actual GPU journal and review the displayed replies. Formatting recovery is
not semantic validation. Reusing these inspected cases measures regression
behavior; a fresh task study is needed before a general quality claim.

## Executed V21 comparison

Private saved23 completed and passed frozen replay/token verification: ten
replies/27 calls, no input rejection or truncation, 11.32 GiB peak. Both intended
normalizations occurred. The compute case now renders dated price/hardware
facts; its new draft is filtered for lacking a historical basis, so the reply
still omits the requested arithmetic and update date. The close-reading case
now renders source-backed allegation/prediction statements, but still omits the
requested certainty/inference distinction. Neither case is a complete answer.

The two former parser blocks are gone. Pipeline statuses are six model-approved
and four partial, not a quality score. Four displayed replies differ from V20,
including downstream histories after a changed reply. Masked source/task review of neutral C against B finds partial gains in the
compute and close-reading cases, a regression in the two-document comparison,
and no completed closing minute. Six replies are identical. All twenty B/C
raw-text and legacy response bindings verify. Do not treat parser recovery as
semantic validation. The next independent diagnostic targets
[source-certainty checking](GHOST_CERTAINTY_CALIBRATION.md).
