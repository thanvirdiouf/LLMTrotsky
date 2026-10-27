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
