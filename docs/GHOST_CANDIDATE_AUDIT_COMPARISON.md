# Fixed neutral-source audit comparison

The 27B feasibility probe passed on free Kaggle's two T4 GPUs. The next question
is whether the candidate makes better factual judgments than the archived 9B
control. This experiment tests that narrow question before attempting fresh
source synthesis or sustained conversations.

## Verified prerequisite

`ghost_candidate_probe_v3`, private saved38, completed one 5,000-input/16-output
generation with finite logits. Peak allocations were 10.59 GiB on GPU0 and
9.77 GiB on GPU1. All six synthetic attention checks passed; all 256 actual
full-attention calls used the specified efficient backend. Exact frozen payloads
and local tokenizer reconstruction verified. This is memory and numerical
evidence for that workload, not source-quality evidence or a guarantee for
longer generations. See [the attention experiment](GHOST_EFFICIENT_ATTENTION_PROBE.md).

The user explicitly approved the 27B probe and bounded tests. They subsequently
clarified that asking why 27B was being tested reflected forgetting the earlier
discussion, rather than cancelling permission. Existing private/free limits
remain in force. Unsloth is a possible separate comparison, not part of this run.

## Frozen comparison

The baseline is `ghost_smoke_v30`, private saved32. It used ten neutral-source
audit candidates in five valid/invalid pairs. Its fixed development scores were
four correct approvals, one correct rejection, one false approval, one false
rejection and three format blocks; zero pairs had both decisions correct.

The new `ghost_candidate_audit_v1` bundle contains the exact ten archived model
messages: identical source packets with authorship, speakers, dates, annotations
and context; identical questions; identical candidate statements; identical
auditor instructions. No retrieval or prompt rewriting occurs. The candidate
tokenizer reconstructs the same input IDs for every control case, ranging from
1,893 to 1,913 tokens. Both use the archived complete greedy generation
configuration, thinking disabled, and a 1,024-token output cap.

Expected decisions, rationales and pair annotations remain local, hash-bound to
the original evaluation specification. They are not in the model input. Candidate
statements intentionally include unsupported assertions; they are test data and
never become chatbot answers or accepted memory.

The candidate is the unchanged pinned `Qwen/Qwen3.5-27B` revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`, NF4 double quantization with FP16
compute. It uses the exact successful probe's placement and efficient-attention
helpers: 34 decoder layers on GPU0, 30 on GPU1, no CPU/disk parameter offload,
and a 13 GiB weight budget per GPU. The 9B baseline used a different preparation
path and default SDPA. This is a whole-system comparison, not an experiment that
isolates model size or attention implementation alone.

## Runtime and verification

The separate runtime first checks the six synthetic attention cases, then loads
the pinned model. It makes one unscored 5,000-input/16-output memory probe and
then at most ten evaluation calls. Each call checks finite logits without
modifying scores, and records input/output tokens and full-attention coverage.
The installation cap is 600 seconds; the workload cap is 2,700 seconds. Partial
failure is recorded, with no automatic retry or journal resume. Existing output
artifacts block a second run. No server is started.

Build and verify with the CPU tokenizer environment:

```bash
.venv-tokenizer/bin/python scripts/build_ghost_candidate_audit.py --version ghost_candidate_audit_v1
.venv-tokenizer/bin/python scripts/verify_ghost_candidate_audit.py --version ghost_candidate_audit_v1 --tokenizer evaluation/ghost_candidate_audit_v1/tokenizer_assets
```

After completion, download the exact saved version's
`ghost_candidate_audit_bundle/` and `ghost_candidate_audit_results/` into
`evaluation/runs/ghost_candidate_audit_v1/download/`. The verifier checks frozen
payloads, preserved archives, model identity, packages, generation settings,
placement, memory, finite logits, per-call and total backend coverage, journal
hashes, stops and local token reconstruction. It then replays the archived 9B
parser and coverage behavior in a separate CPU process against the new outputs.
The replay uses the old model identity only inside the old parser harness;
actual candidate runtime/results are explicitly labelled 27B.

Scoring distinguishes correct approval, correct rejection, false acceptance,
false rejection, format/truncation blocks, and unscored incomplete judgments.
A format block is not a correct semantic rejection. Pair coverage is reported
separately. CPU preflight reproduced all ten old control outcomes, including its
three blocks, using the archived outputs; this involved no new inference.

Final judgments also require direct source review of the actual wording,
certainty, speaker roles and unsupported details. The implementing coding
agent's review is unblinded development review. The fixed expected decisions
were authored during development; they are not independent or human gold.
Model approval is not independent confirmation of its own source fidelity.

Five new tests cover scope and no-resume guards, frozen input/expectation
separation, preservation of the successful GPU helpers, token/stop/logit/backend
checks, complete-result verification, and distinct scoring outcomes. All 116
focused ghost tests pass. Full389 tests retain the two known historical
`evaluation/README.md` snapshot errors. Historical locks were not changed.

## Decision after results

No result automatically replaces active 9B/v3 or enables a checker. A failed job
requires diagnosis. Poor or inconsistent factual judgments leave the source
inspection path and existing baseline available; they do not justify training.
If the comparison shows useful source-faithful judgments, the next step is a
bounded synthesis experiment with reviewed sources, then the already fixed
transfer questions if synthesis quality warrants it. Sustained conversation
and voice evaluation follow usable factual synthesis. Training and both adapters
remain paused/experimental throughout.
