# Corpus and dataset quality report

Snapshot: 2026-10-22 18:00 IST.

## Inventory

| Artifact | Verified count/status |
|---|---:|
| Downloaded HTML pages | 1,913 |
| Request failures / pending URLs | 0 / 0 |
| Text / index / directory-placeholder pages | 1,823 / 53 / 37 |
| Cleaned candidate words, including duplicates and unreviewed material | 5,666,449 |
| Unreviewed authentic-dialogue candidates | 1,606 |
| Provisional automatic source families | 1,482 |
| Near-duplicate or excerpt links | 58 |
| Coding-agent-reviewed seed examples / source documents | 500 / 123 |
| Reviewed multi-turn conversations | 65 |
| Initial evaluation cases | 36 |
| Training-seed / evaluation family overlap | 0 |
| Model responses, adapters trained, or paid generation calls | 0 |

This is the available linked archive within the configured scope, not proof of the complete lifetime works. Nine explicitly listed supplemental works are included. No PDF extraction or general crawl of MIA's other archives was performed.

## What was verified

The complete offline pipeline ran successfully after the expansion. End-to-end validation verified all 1,913 cached HTML hashes, all 1,606 candidate answer spans, seed/evaluation evidence hashes, seed composition-date eligibility, whole-family reservations, the chat export, and logged text repairs. The previously run unit suite has **68 passing tests**, including mocked remote-client behavior and rejection of invented evidence, held-out generation inputs, automatic teacher approval flags, forged/self-consistent source records, stale holdout manifests, changed date evidence and malformed conversation roles. Full-message and persona hashes protect the frozen review; duplicate record IDs are rejected.

All original HTML and source text remain available. The supplied `trotsky_corpus.txt` is unchanged; its SHA-256 is `2d52075747e708b8cc8924d9a1b949be05098db34a8a4260967410280f443918`.

## Cleaning performed

HTML structure restores paragraph and inline word boundaries that were lost in the original concatenation. Metadata, navigation, explicitly identified editorial regions, notes, stage directions, and authored-body candidates are separated. Conservative overlays repaired 16 malformed punctuation entities and two reviewed OCR errors. Every change is recorded in `corpus_mia/text_repairs.jsonl`.

The Malaquais essay’s unmarked end of editorial introduction was repaired through an exact source-hash and paragraph-prefix exception, preserving the 1941 editorial introduction separately. This is a reviewed source-specific boundary, not a generic rule for other pages.

No broad semantic rewriting, speculative word-boundary reconstruction, or unlogged deletion of repeated passages was performed. Candidate text still requires review where source authorship, OCR, quotations, or editorial boundaries are ambiguous.

## Historical date boundary

The working persona is Mexico, **19 August 1940**, an adjustable implementation assumption. Explicit composition evidence puts 597 documents on/before that date, 2 after it, and 4 across an uncertain interval overlapping it. 1,310 documents need composition-date review or additional evidence; URL years alone are not promoted to verified dates. This pass recovered explicit metadata or signed-dateline evidence for 53 more documents and widened or refined 21 existing intervals. Three source-date conflicts remain explicit: the August letters with metadata/body disagreement retain ranges rather than a falsely precise day. Unsigned headings, publication dates, quoted letters and other authors’ signatures are excluded as composition evidence.

The article dictated on 20 August and the three letters written that day are outside this persona's boundary. Conversely, a journal written in 1936–1937 and published in 1941 can supply pre-cutoff authored material. Later editorial introductions remain a separate authorship/time issue.

## Seed and evaluation limits

The seed has 82 verbatim transcript answers, 132 exact prose passages and 286 labelled adaptations from 123 source documents across 111 families. It includes 65 multi-turn conversations and 86 hearing responses, including 53 from five previously unused, explicitly dated sessions in the same hearing source family. Final assistant answers include 357 short, 137 medium and six long responses. The 500-example expansion target is complete. This remains a bootstrap set, not a historian-validated gold dataset or the full 2,000–5,000-example pilot. The audit corrected two context omissions in the original seed: the explanation of the N initial now includes Natalia, and the 1917 relations question now names its pre-August period. Source proposals, baseline audit findings and resolution are retained in `reviews/`. Some historical sources contain partisan judgments and factual mistakes; preserving a speaker's words does not establish every claim as true.

The evaluator has six source-backed test cases from the reserved My Life family, six validation cases from The Revolution Betrayed, and 24 behavioral cases. Entire grouped families are reserved: 48 test documents and 14 validation documents. Family grouping can miss low-overlap translations or excerpts. These are holdouts from this project's SFT, not proof that the base model never saw the sources in pretraining.

No empirical model-quality scores are available. The remote client has been tested with mocks, request previews and real-corpus rejection of an unreviewed passage; no inference service has been called. It still requires a configured service and live compatibility smoke test. Its generated outputs remain pending source review.

## Next priorities

1. Continue passage-level authorship, date, OCR and context review using `corpus_mia/review_queue.jsonl` and the source-family manifest. Keep future seed growth balanced across genres and source families.
2. Prepare a remote prompt-only baseline and verify chat templates, label masks and architecture compatibility before any LoRA run.
3. Run prompt-only baseline evaluation on a remote service, then a small remote LoRA experiment after checking Qwen3.5 templates, label masks and architecture support.
4. Compare retrieval variants when baseline failure analysis identifies where source access helps.
