# Checker diagnostic after saved41

Saved41 resolved quotation formatting blocks but still displayed an incorrect
resident-secretary replacement after checker approval. Its checker explanation
also confused Trotsky's 1887-to-1897 self-correction with Goldman's retraction of
1902. These are source-interpretation failures with the needed passages already
present. See [the synthesis review](GHOST_CANDIDATE_SYNTHESIS.md).

The next experiment, `ghost_candidate_audit_v2`, tests two focused hearing
failure pairs plus three familiar calibration pairs. It is a post-hoc
development diagnostic, authored by the implementing agent after reading the
failures. It is not independent or human validation, and does not consume the
fixed fresh-transfer questions.

## Cases and privacy

The tracked blueprint is `config/ghost_generated_audit_design_v1.json`.
One negative candidate is the actual saved41 resident-secretary paragraph,
loaded by case, segment and UTF-8 hash from ignored output. Its positive partner
states that the resident count is not established and distinguishes it from the
total-secretary range and the nonresident Russian secretary. Generated text is
not copied into Git.

The other hearing pair expresses the checker reason's date confusion as a
plain factual claim, then contrasts it with a source-reviewed correction that
distinguishes Trotsky's self-correction, Goldman's opening statement, and the
1902 article. This pair is authored from the observed failure; it is not a
verbatim replay of an entire displayed answer.

The remaining pairs preserve the old first-name, hypothesis and informant-versus-
subject candidates, questions and expected decisions from v30. All ten cases
receive the same complete hearing, Rakovsky and Bessedovsky packets. These
familiar controls therefore have added hearing context, and cannot be presented
as identical prompts to the original two-source calibration. The neutral checker
system instruction remains identical to v30.

The expected decisions, rationales and pair structure remain in a local frozen
evaluation specification. Model messages contain sources, questions and
candidate statements only. Runtime case identifiers do not encode validity.
The original generated paragraph and all new outputs remain ignored locally.

## Runtime and verification

The pinned 27B candidate, shared runtime, GPU placement helper, efficient
attention helper and 5,000-input/16-output probe are byte-identical to saved41.
There are ten independent evaluation calls capped at1,024output tokens plus
one unscored memory probe. The same private notebook, free two-T4 target,
600-second installation cap and2,700-second workload cap apply. No training,
adapters, paid compute, public uploads, CPU/disk parameter offload or active9B/v3
replacement is allowed.

```bash
.venv-tokenizer/bin/python scripts/build_ghost_candidate_audit.py --version ghost_candidate_audit_v2 --generated-diagnostic
.venv-tokenizer/bin/python scripts/verify_ghost_candidate_audit.py --version ghost_candidate_audit_v2 --tokenizer evaluation/ghost_candidate_audit_v2/tokenizer_assets
```

After completion download the exact saved version's `ghost_candidate_audit_bundle/`
and `ghost_candidate_audit_results/` to
`evaluation/runs/ghost_candidate_audit_v2/download/`.
The verifier reconstructs the hash-bound diagnostic design and source requests,
checks exact payloads, tokens, generation, scope, numerical health and GPU
coverage, then replays archived neutral-audit parsing/coverage behavior in an
isolated CPU process. The direct parser harness avoids the old matched-retriever's
two-packet limit; it adds no retrieval or changed source interpretation.
CPU preflight reproduces every old v30 outcome and saved39's five correct pairs
with that parser, including distinctions between blocks and semantic rejection.

New hearing pairs and familiar control pairs are reported separately. Inputs
are3,574–3,666tokens, within the unchanged cap. Three new tests cover private
candidate hash binding, preservation of the old pairs, source-context equality,
withheld labels, parser outcomes and separate group reporting. All124 focused
tests pass; full397 retains the same two historical README snapshot errors.
Original locks were not changed.

## What the result can justify

Review actual final reasons against the original passages as unblinded
implementation review. Score false approval, false rejection, truncation/format
blocks and incomplete judgments separately. A later qualification does not
automatically repair an incorrect proposed field.

The new questions explicitly focus on the failed distinction; success would
show the model can make that focused judgment. It would not repair the original
synthesis or establish reliable checking under its broader question. It could
justify testing task decomposition on the same synthesis questions. Failure on
the focused pairs would further limit reliance on the self-checker. Neither
outcome permits automatic activation, training or fresh transfer. Fresh transfer
continues to depend on actual source-faithful, useful synthesis.
