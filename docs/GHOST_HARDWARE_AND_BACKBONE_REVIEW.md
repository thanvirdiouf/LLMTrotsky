# Hardware inventory and possible backbone comparison

The thinking diagnostic V31/saved33 completed and passed frozen replay/token
verification: ten calls, nine truncated generations and one complete judgment.
The sole final judgment correctly rejects a claim of independent verification.
The nine output-budget blocks are not semantic rejections. Each truncated call
used4,096 tokens; calls took roughly5.6–5.8minutes each. Peak memory stayed11.32GiB.
This does not establish general incapacity of the model, but this inference
policy is unusable as the proposed automatic gate under the tested budget.
No reader integration, fresh transfer or training follows.

The next useful preparation is a hardware-only inventory of the same existing
private free Kaggle notebook. `ghost_hardware_v1` loads no model, downloads no
weights, makes zero generation calls and installs nothing. Internet is disabled.
It records GPU count/name/free and total VRAM, available host RAM, working-disk
space and software versions. No credentials, environment variables, hostnames
or source passages are collected. Its notebook, metadata and source are frozen;
verification compares the downloaded source and checks inventory consistency.

```bash
python3 scripts/build_ghost_hardware_probe.py
# After the exact saved version has been downloaded:
python3 scripts/verify_ghost_hardware_probe.py --version ghost_hardware_v1
```

An inventory is not evidence that a larger model will fit. Any future model load
needs a separate bounded memory/placement probe and an abort on insufficient
resources. The current9B preparation helper requires a single CUDA device and
cannot silently be reused for a distributed model. Frozen historical helpers
and adapters must remain unchanged.

A reviewable candidate proposal is in
`config/ghost_backbone_comparison_proposal_v1.json`. Public metadata was read for
[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), pinned to revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`; its card lists Apache2.0 and Transformers
support. No candidate weights have been downloaded and no larger-backbone run
has been submitted. This is a candidate experiment, not a promise of improved
source fidelity or a product recommendation based on model size alone.

The proposed first gate is free-Kaggle feasibility, no scored answers and one
short memory probe. Only if actual placement/memory works would a separate
frozen run apply the same ten factual calibration candidates and expectations.
It permits at most ten judgment calls plus one memory probe, no training,
adapters, paid compute or active-pointer change. It would compare whole systems,
including different model/runtime placement, rather than isolate capacity alone.
The hardware inventory and a clear backbone-scope decision precede that work.
The existing active v3 and source-inspection commands remain available.

Four tests exercise immutable notebook inputs, private/offline target settings,
no model/training calls in probe code, local-execution rejection, simulated
hardware output, exact downloaded-source binding, inconsistent or unexpected
GPU inventory, and the explicit absence of a model-fit claim.

All101 focused tests pass; full374 retains the same two historical README
snapshot errors. No protected lock or archived adapter has been changed.

The hardware-only probe was submitted as private Kaggle saved34 after e156d54
was committed and pushed. Inventory results are pending; no larger model has
been loaded or run.

## Verified inventory and cache-filesystem follow-up

Saved34 verifies two Tesla T4 devices, each with14.56GiB total and14.46GiB free
at probe time. The two GPUs provide about28.9GiB free in aggregate, not one
contiguous GPU allocation. Earlier model runs used GPU0 only. Available host RAM
was29.65GiB. The working output filesystem had19.50GiB free.

Public file metadata at the proposed27B revision totals55,563,022,432bytes across
11safetensors shards, about51.75GiB before tokenizer/cache overhead. No weights
were downloaded. They would not fit the working output filesystem. However,
the existing loader uses Hugging Face's default cache location rather than
`/kaggle/working`, so the working-disk reading alone cannot establish download
feasibility or justify paid compute.

The next hardware-only `ghost_hardware_v2` therefore records the actual SDK
cache filesystem, home and temporary filesystem capacities and whether each is
the same filesystem as the working directory. It reports no actual path or
environment variable value. If the SDK is unavailable, the fallback is explicitly
labelled as unverified rather than claimed to be the real configured cache.
It still disables internet, downloads no model, makes zero generation calls and
changes no active route. The verifier accepts frozen v1 inventory as before and
requires the additional storage fields for schema2.

Build new source as `--version ghost_hardware_v2`; do not rebuild frozen v1 with
changed source. Its memory/cache measurements still cannot prove model fit:
a larger-model experiment would require correct sharding and a real bounded
load/probe before scoring. A new test checks storage schema consistency while
preserving backward verification of saved34.

Current checks:102 focused tests pass; full375 retains the same two historical
README snapshot errors. The updated verifier also re-verifies the actual saved34
source and inventory unchanged.
