# Separate pinned candidate feasibility probe

The cache inventory from private saved35 is verified. It reports two T4 GPUs
with about28.9GiB aggregate free VRAM and an SDK-resolved model-cache filesystem
with about1.02TiB reported free. The cache is a different filesystem from the
19.5GiB working output volume. The original27B shards total51.75GiB, so the
reported cache headroom clears the download-size check. None of these figures
proves that a sharded model or its activations will fit.

The next experiment is a separate feasibility candidate, leaving the active
9B pin, source plan, archived helpers and adapters unchanged. The initial automatic approval review rejected submission because the earlier
permission covered pinned9B. Nothing was submitted. On27October2026 the user
explicitly approved the prepared27B probe and bounded27B inference follow-ups
if it fits, retaining the10reply/30call-plus-probe limits, private/free target,
no paid hardware, training, adapters or active9B replacement.

Candidate: [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`. Only public tokenizer/config assets have
been downloaded locally. The candidate's weights will be fetched at the frozen
revision on Kaggle; no local GPU or model-weight download is used.

The frozen `ghost_candidate_probe_v1` notebook:

- uses the same private notebook and pinned software stack on two free T4s;
- loads NF4 double-quantized weights with FP16 compute and SDPA, allocating at
  most13GiB of planned weight memory per GPU;
- requires all frozen parameters to reside on GPU0/GPU1 and rejects CPU/disk
  offload or adapters;
- makes one unscored5,000-input-token memory/placement probe, at most16 output
  tokens, with thinking disabled;
- checks raw logits for finiteness and records peak allocation on each GPU;
- generates no source answers, quality scores or training records;
- caps package installation at10minutes and probe runtime after setup at45minutes,
  then exits. It runs no server and performs no retry cascade.

This is a fresh inference-only sharded loader. It does not modify or call the
archived single-GPU PEFT-preparation helper. A future quality comparison must
acknowledge differences in model size and preparation/runtime rather than call
it a single-variable capacity experiment. The successful result would establish
only this exact probe's memory/placement/numerical-health behavior. It would not
validate factual reasoning, longer outputs, concurrency or the chatbot persona.

The builder freezes notebook, metadata, source, tokenizer/config hashes and exact
probe token IDs. The verifier compares downloaded payloads, the model revision,
resource limits, parameter devices, token reconstruction and GPU peaks; it also
checks the protected historical archive. A blocked result remains a failed
feasibility result, even if Kaggle marks the finite notebook execution complete.

```bash
.venv-tokenizer/bin/python scripts/build_ghost_candidate_probe.py
# After exact-version download:
.venv-tokenizer/bin/python scripts/verify_ghost_candidate_probe.py \
  --version ghost_candidate_probe_v1 \
  --tokenizer evaluation/ghost_candidate_probe_v1/tokenizer_assets
```

Three tests cover resource/placement rejection, no offload/trainable/adapted
parameters, local-inference rejection, exact frozen inputs, one-call limits,
wrong model or outputs, finite-logit checks, GPU peaks, blocked-result handling
and the unchanged active9B constant. CPU preflight reconstructs the5,000 tokens
and passes the reported inventory through the resource gate. GPU fit remains
unestablished until the actual probe passes.

After a verified pass, the next bounded quality experiment would use the same
ten fixed neutral-auditor candidates and expectations, no more than ten judgment
calls plus a short memory probe. Only useful source-fidelity results would
justify the existing fresh transfer questions, followed by sustained dialogue.
No result in this chain automatically activates a new model or resumes training.

All105 focused tests pass; full378 retains the same two historical README
snapshot errors. CPU preflight reconstructs exactly5,000 candidate-tokenizer
input tokens, validates frozen limits and passes measured hardware through
the resource gate. No local model weights have been loaded or downloaded.

After explicit user approval for the probe and bounded27B follow-ups, the exact
prepared candidate probe was submitted as private Kaggle saved36. The first
automatic-review rejection did not submit a workload; the approved submission
succeeded. Results are pending.

## Verified v1 result and targeted placement test

Candidatev1/saved36 completed in about9.1minutes in the probe program, including download/loading. Exact payload,
model revision and scope verification passed, but the memory probe failed:
weights loaded across both GPUs, then the5,000-token generation ran out of
memory on GPU1 while requesting574MiB. About111MiB was free there. No source
quality was evaluated, and no successful fit is claimed.

The recorded automatic map assigned29 decoder layers plus embeddings to GPU0,
and35 layers plus the output head to GPU1. This suggests a placement-headroom
hypothesis worth testing before changing hardware or context limits; the first
probe did not record enough per-GPU allocation detail to prove the cause.

Candidatev2 fixes the64-layer map at34layers on GPU0 and30 on GPU1, keeping the
embedding and output head on their original respective GPUs. This moves five
layers off the device that ran out of memory. All5,000 input tokens,16 output
cap,13GiB per-GPU weight budget, packages, model revision and runtime limits are
unchanged. The explicit placement additionally checks actual allocated memory
after loading against the same weight budget. It records per-GPU allocation and
reservation after loading and on failure, so another block can be diagnosed.
No input is removed, no scored questions are run, and there is no CPU/disk
fallback. The old automatic-map run remains frozen and verifiable.

Two additional tests check the exact34/30 map, architecture guard, unchanged
probe/model/budgets, and verifier rejection of a wrong split or exceeded weight
budget. CPU preflight confirms the v2 probe is byte-identical to v1. This is one
targeted feasibility test of an observed failure, not a blind retry or an
increase in resource/output allowances.

Current checks:107 focused tests pass; full380 retains the same two historical
README snapshot errors. The updated verifier re-verifies the actual saved36
blocked result. Candidatev2 CPU preflight confirms byte-identical probe inputs
and unchanged model, software and resource limits.
