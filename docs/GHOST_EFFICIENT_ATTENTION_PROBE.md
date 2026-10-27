# Candidate-only efficient-attention feasibility test

Candidatev2/saved37 is exact-payload verified and blocked during generation.
Weights occupied about8.60GiB on GPU0 and7.86GiB on GPU1 after loading. The
5,000-token forward failed on GPU0 requesting another574MiB. GPU0's recorded
peak was13.96GiB, while GPU1 had not yet reached substantial forward allocation.
The redistribution moved the failure; it did not establish a successful fit.
No source answers or model-quality results were produced.

The next change targets a source-guided attention-memory hypothesis rather
than another placement guess. The pinned Transformers5.20.2 SDPA adapter chooses
native grouped-query attention for unmasked256-dimensional keys/values. PyTorch
2.11 documents native GQA support in Flash and math kernels, while its CUDA
source requires SM80 or newer for Flash. The measured T4s are SM75. This makes
math attention and its large intermediate matrices a likely contributor to the
observed transient allocations, although the previous runs did not trace the
actual kernel.

Sources: [PyTorch2.11 SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html),
[PyTorch2.11 CUDA dispatch source](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp).
Local inspection used the installed, matching Transformers5.20.2 sources.

Candidatev3 retains the same34/30 layer map, model pin, input bytes,5,000 input
and16 output tokens,13GiB per-GPU weight limit, packages and timeouts. A separate
candidate attention adapter expands the KV groups explicitly and permits only
PyTorch's efficient-attention kernel. Its module proxy disables the native-GQA
flag for the library call without changing the model's group metadata. There is
no silent math-kernel fallback. Dynamic-cache updates remain in the model, as
before; paged-cache use is explicitly rejected.

The adapter delegates masking and causality to the existing HF SDPA function
and registers the same SDPA mask formatter under the new backend name. Both
registrations matter: otherwise a custom attention backend can lose masking.
See [Hugging Face attention backend documentation](https://huggingface.co/docs/transformers/attention_interface).
No default backend or active9B process is modified.

Before downloading model weights, the notebook checks small FP16 attention
examples on both T4s against the math reference: causal prefill, one-token decode
with prior keys, and masked prefill. All six cases must meet absolute tolerance
0.002 and relative tolerance0.01 and produce finite output. These are synthetic
kernel checks, not model generation calls or source judgments. If the efficient
kernel is unsupported or differs numerically, the probe stops before fetching
the large checkpoint. Floating-point differences between kernels remain
possible; these small checks do not prove equivalence for every input.

The actual model probe must still have finite logits. Runtime counters must show
that every full-attention layer used the new backend for every generated token.
The verifier requires the six kernel checks, matching mask policy, disabled
native GQA and complete call coverage. Load and failure allocation diagnostics
remain available, with traceback frame names if blocked. Existing result files
cannot trigger an automatic second probe call.

Four new tests cover KV expansion, proxy/module and mask preservation,
efficient-only context, both registry entries, no default-backend replacement,
cache/head-shape rejection, frozen policy/helper inclusion, verifier rejection
of missing checks/coverage and rerun prevention. CPU preflight proves the probe
is byte-identical to v2 and only the attention policy/diagnostics change. Actual
CUDA equivalence and full-context memory success are pending GPU execution.

A success would justify the bounded factual-auditor experiment already approved
by the user. It would not establish source fidelity or activate the new model.
Training, adapters, paid compute and the active9B setup remain unchanged.

All111 focused tests pass; full384 retains the same two historical README
snapshot errors. The updated verifier re-verifies the actual saved37 failure.
Candidatev3 CPU preflight reconstructs exactly5,000 tokens, with byte-identical
probe data and unchanged model, map, packages and resource/time caps. CUDA
backend equivalence is explicitly pending the real notebook execution.
