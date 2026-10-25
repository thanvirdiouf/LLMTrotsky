"""Prepare a quantized base without transient FP32 embedding copies on the GPU.

This module imports no training libraries. Call it before attaching adapters.
CPU placement is temporary preparation staging; all parameters must be restored
to the selected GPU before the helper returns or any model computation runs.
"""


def prepare_frozen_quantized_base(model, *, dtype, device, prepare_kbit,
                                  training, stage_callback=None):
    """Stage frozen input/output matrices on CPU during PEFT's global upcast.

    PEFT also upcasts the large nonquantized embedding and output matrices. Move
    both to host memory first, run the standard preparation unchanged, downcast
    those two matrices while still on CPU, and then restore their GPU placement.
    Norms and the other nonquantized stability-sensitive parameters retain PEFT's
    FP32 policy. On failure the caller receives the original exception; a partly
    staged model must never be used for a forward pass.
    """
    if not str(device).startswith("cuda:"):
        raise ValueError("Preparation requires one explicit CUDA device index")
    embeddings = []
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is None:
            raise RuntimeError("Input/output embeddings must exist")
        if all(module is not previous for previous in embeddings):
            embeddings.append(module)
    for module in embeddings:
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.to(device="cpu")
    if stage_callback:
        stage_callback("embedding_head_staged_on_cpu")

    prepared = prepare_kbit(model, use_gradient_checkpointing=training,
                           gradient_checkpointing_kwargs={"use_reentrant": False})
    if prepared is not model:
        raise RuntimeError("PEFT preparation unexpectedly replaced the base model")
    if any(parameter.requires_grad for parameter in prepared.parameters()):
        raise RuntimeError("PEFT preparation must freeze every base parameter")
    if stage_callback:
        stage_callback("nonquantized_parameters_prepared")

    for module in embeddings:
        # Separate operations matter: the FP32 -> compute conversion must happen
        # on CPU, before allocating the restored matrix on the GPU.
        module.to(dtype=dtype)
        if any(str(parameter.device) != "cpu" for parameter in module.parameters()):
            raise RuntimeError("Embedding/head dtype conversion escaped CPU staging")
        module.to(device=device)
    wrong_device = [name for name, parameter in prepared.named_parameters()
                    if str(parameter.device) != str(device)]
    if wrong_device:
        raise RuntimeError(f"Base parameters remain outside {device}: {wrong_device[:16]}")
    prepared.config.use_cache = False
    if stage_callback:
        stage_callback("all_base_parameters_restored_to_gpu")
    return prepared
