# Research archive

The working tree contains the current app, source pipeline, transfer design,
and tools for the recent conversation experiments. The complete pre-cleanup
code, documentation, configurations, and tests are preserved at Git commit
`8f99d91cd0e33ba7201bfd6fbe2a87d6c2c54552`.

Read the original handoff or a specific experiment note:

```sh
git show 8f99d91:PROJECT_HANDOFF.md
git ls-tree --name-only 8f99d91:docs
git show 8f99d91:docs/CONVERSATION_RETRIEVAL.md
```

## Local artifacts

The private archive is `.research-archive/local-research-2026-11-10.tar.gz`.
It holds 4,502 files in 739 MiB, including superseded sources, extracted data,
reviews, notebook bundles, run outputs, and training artifacts. Paths match
their original locations. The active source
packets, recent conversation results, both final adapters, and files required
by current verification remain available in place.

List or recover individual artifacts into a separate directory:

```sh
sha256sum -c .research-archive/local-research-2026-11-10.tar.gz.sha256
tar -tzf .research-archive/local-research-2026-11-10.tar.gz
mkdir -p ../LLMTrotsky-restored
tar -xzf .research-archive/local-research-2026-11-10.tar.gz \
  -C ../LLMTrotsky-restored --keep-old-files -- PATH_FROM_LIST
```

To restore the full historical workspace, run the following from the current
checkout. This combines the old tracked tree, the private files still in use,
and the compressed archive in a separate checkout.

```sh
git worktree add --detach ../LLMTrotsky-research 8f99d91cd0e33ba7201bfd6fbe2a87d6c2c54552
git ls-files --others --ignored --exclude-standard -z -- \
  config corpus_mia datasets evaluation notebooks reviews training trotsky_corpus.txt \
  | tar --create --file=- --null --no-recursion --files-from=- \
  | tar --extract --file=- --directory=../LLMTrotsky-research --keep-old-files
tar -xzf .research-archive/local-research-2026-11-10.tar.gz \
  -C ../LLMTrotsky-research --keep-old-files
```

The archive and its checksum are local. Back them up together under the same
access controls as the research data. The historical test suite already had two
pilot tests fail on a frozen `evaluation/README.md` hash. The archived files
preserve that state.

## Recent evaluation

With the local tokenizer environment and the saved artifacts, verify saved50:

```sh
.venv-tokenizer/bin/python scripts/verify_review_chat.py \
  --version ghost_review_chat_v6 \
  --tokenizer evaluation/ghost_review_chat_v6/tokenizer_assets
```

Use `ghost_review_chat_v5` in both places for saved49. The verifier checks
frozen bytes, tokenizer, replay, and run scope. Source judgments are recorded
separately in each run’s `SOURCE_REVIEW.json`. The verifier reads archived
protected scripts from the pre-cleanup Git revision when they are absent
from the checkout.
