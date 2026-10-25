# Seed dataset

- `train_seed.jsonl`: chat messages for a small initial experiment, using the configured historical persona.
- `seed_records.jsonl`: the same examples with exact source URLs, HTML hashes, source-block IDs, original evidence, date evidence, source families, and review notes.
- `seed_preview.md`: readable questions and answers.
- `seed_audit.json`: counts and validation status.

The current seed contains **500 examples from 123 source documents across 111 source families**: 82 verbatim transcript answers, 132 exact prose passages, and 286 explicitly labelled source-grounded adaptations, including 65 multi-turn conversations. It spans letters, interviews, discussions, speeches, journalism, journals, literature and political essays; 86 examples are hearing-based, drawn from multiple sessions in one source family. The final assistant answers include 357 short answers (up to 40 words), 137 medium answers and six answers over 120 words. This is a reviewed bootstrap set, not the full 2,000–5,000-example pilot or independent historian validation. No separate LLM API or local model inference was used to author it.

The review specifications are in `reviews/seed_specifications.json`; the evidence lock fails rebuilding if reviewed evidence, messages, date evidence or the configured persona changes silently. To deliberately update a review, inspect the source and use `--review-update-note` with a concrete explanation. Earlier locks are retained in the review history. The same note can be passed to `scripts/prepare_corpus.py --review-update-note "Concrete source review reason"` for a complete offline rebuild.

`verbatim_passages` preserves exact prose answers under the `verbatim_source` origin. `adapted_conversations` contains reviewed alternating user/assistant turns, with all assistant claims checked against the selected source evidence. These are authored conversational adaptations, not historical dialogue transcripts. The preview renders every turn, and message hashes protect the full exchange.

The compiler checks current whole-family evaluation reservations before writing the seed. The remote generation runner accepts only the exact frozen passage selections; approval of one passage never approves the rest of its document.

Do not train directly on `corpus_mia/documents.jsonl` or `qa_candidates.jsonl`: these retain unreviewed source material. Extend the approved set only after attribution, date, context, evidence and split checks. Reserve evaluation families before generating more pairs.

GPU training, chat-template/token-mask validation, and empirical baseline comparison remain remote-work stages. No adapter has been trained.
