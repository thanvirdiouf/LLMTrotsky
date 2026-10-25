"""Kaggle-only QLoRA preflight; heavy dependencies are imported only by run_smoke.

The notebook bundles this module, training_masks.py, the reviewed seed, and its lock.
This uses explicit labels and a short PyTorch loop so no trainer can re-tokenize or
replace the inspected assistant masks. It is a compatibility check, not a quality run.
"""
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_SETTINGS = {
    "model_id": "Qwen/Qwen3.5-9B",
    "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "seed": 42, "max_length": 1024, "steps": 2,
    "micro_batch_size": 1, "gradient_accumulation_steps": 1,
    "learning_rate": 0.0001, "weight_decay": 0.0, "max_grad_norm": 1.0,
    "amp_initial_scale": 1.0,
    "rank": 8, "lora_alpha": 16, "lora_dropout": 0.0,
    "optimizer": "torch.optim.AdamW", "packing": False,
    "truncation": "drop_whole_example", "gradient_checkpointing": "non_reentrant",
    "quantization": "nf4_double", "attention_implementation": "sdpa",
    "smoke_max_new_tokens": 64,
    "smoke_question": "Please introduce yourself in two sentences.",
}

PROJECTIONS = {
    "self_attn": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "linear_attn": {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"},
    "mlp": {"gate_proj", "up_proj", "down_proj"},
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def configure_probe_stop(tokenizer, generation):
    """Use the tokenizer's actual assistant boundary for this operational probe.

    The frozen inference baseline's config EOS differs from its tokenizer's
    im_end ID. Preserve that historical comparison in pilot_lock.json; the smoke
    probe must instead stop on the same boundary its training labels supervise.
    """
    answer_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (type(answer_end_id) is not int or answer_end_id < 0
            or tokenizer.convert_ids_to_tokens(answer_end_id) != "<|im_end|>"):
        raise RuntimeError("Tokenizer has no valid <|im_end|> assistant boundary")
    original = generation.eos_token_id
    original_ids = [original] if isinstance(original, int) else original or []
    generation.eos_token_id = answer_end_id
    return {"policy": "tokenizer_im_end_for_operational_smoke_only",
            "original_config_eos_token_id": original,
            "operational_eos_token_id": answer_end_id,
            "differs_from_original_config": original_ids != [answer_end_id],
            "quality_comparison": "Fresh baseline with matched stop IDs is required before pilot quality comparison"}


def select_targets(module_names, layer_types):
    """Require every expected text projection and exclude vision/head/embeddings."""
    expected = set()
    for i, kind in enumerate(layer_types):
        if kind not in {"linear_attention", "full_attention"}:
            raise ValueError(f"Unexpected decoder layer type: {kind}")
        attention = "linear_attn" if kind == "linear_attention" else "self_attn"
        for group in [attention, "mlp"]:
            expected.update(f"model.layers.{i}.{group}.{leaf}" for leaf in PROJECTIONS[group])
    actual = set(module_names)
    missing = expected - actual
    if missing:
        raise ValueError(f"Expected text Linear projections missing: {sorted(missing)}")
    # Fail on architecture drift, rather than silently missing new decoder projections.
    decoder = {n for n in actual if re.match(r"^model\.layers\.\d+\.", n)}
    if decoder != expected:
        raise ValueError(f"Unexpected decoder Linear projections: {sorted(decoder - expected)}")
    return sorted(expected)


def validate_settings(settings):
    if set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Settings keys must match the smoke-test schema")
    for key in ["steps", "max_length", "rank", "lora_alpha", "smoke_max_new_tokens"]:
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 2 <= settings["steps"] <= 8:
        raise ValueError("This preflight requires two to eight steps covering both conversation types")
    for key in ["micro_batch_size", "gradient_accumulation_steps"]:
        if settings[key] != 1:
            raise ValueError(f"{key} must remain one for this smoke test")
    for key in ["learning_rate", "weight_decay", "max_grad_norm", "lora_dropout"]:
        if not math.isfinite(settings[key]):
            raise ValueError(f"Non-finite {key}")
    if settings["learning_rate"] <= 0 or settings["max_grad_norm"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if not 0 <= settings["lora_dropout"] < 1:
        raise ValueError("Invalid LoRA dropout")
    if (type(settings["amp_initial_scale"]) not in {int, float}
            or not math.isfinite(settings["amp_initial_scale"])
            or settings["amp_initial_scale"] <= 0):
        raise ValueError("amp_initial_scale must be finite and positive")
    fixed = ["model_id", "revision", "optimizer", "packing", "truncation",
             "gradient_checkpointing", "quantization", "attention_implementation"]
    if any(settings[k] != DEFAULT_SETTINGS[k] for k in fixed):
        raise ValueError("Architecture/label/optimizer policy changed; revise the preflight explicitly")


def run_smoke(bundle_dir, output_root, settings=None):
    """Run on one Kaggle GPU, preserving reports on any exception (including OOM)."""
    import datetime
    import gc
    import importlib.metadata
    import subprocess
    import sys
    import time

    bundle_dir, output_root = Path(bundle_dir), Path(output_root)
    cfg = dict(DEFAULT_SETTINGS if settings is None else settings)
    validate_settings(cfg)
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this training preflight on Kaggle; local GPU work is disabled")
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, GenerationConfig, set_seed)
    from huggingface_hub import snapshot_download
    from training_masks import encode_assistant_example, inspect_tokens, collate_examples
    from build_training_pilot import validate_training_payload
    from kbit_preparation import prepare_frozen_quantized_base

    lock_bytes = (bundle_dir / "pilot_lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    payload = (bundle_dir / "train_seed.jsonl").read_bytes()
    rows = validate_training_payload(payload, lock)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator and Internet")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    set_seed(cfg["seed"])
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    packages = {p: importlib.metadata.version(p) for p in
                ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface-hub", "tokenizers", "jinja2"]}
    identity = {"settings": cfg, "packages": packages,
                "pilot_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
                "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "helpers": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in bundle_dir.glob("*.py")},
                "gpu": {"name": torch.cuda.get_device_name(0),
                        "total_bytes": torch.cuda.get_device_properties(0).total_memory,
                        "compute_capability": list(torch.cuda.get_device_capability(0))},
                "dtype": str(dtype), "visible_gpus_used": [0],
                "purpose": "training_compatibility_only"}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / f"smoke-{fingerprint(identity)[:12]}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "identity": identity, "checks": {}, "stage": "tokenization"}

    def write_json(name, value):
        (run_dir / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def checkpoint(stage):
        manifest["stage"] = stage
        write_json("run_manifest.json", manifest)

    def memory(stage):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(0)
        row = {"stage": stage, "allocated_bytes": torch.cuda.memory_allocated(0),
               "reserved_bytes": torch.cuda.memory_reserved(0), "free_bytes": free, "total_bytes": total,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        with (run_dir / "memory.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("Memory:", row, flush=True)
        return row

    try:
        checkpoint("tokenization")
        write_json("pilot_lock.json", lock)
        (run_dir / "pip_freeze.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True))
        # The same immutable revision is used for tokenizer, initial model, and reload.
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
        template_hash = hashlib.sha256(tok.chat_template.encode()).hexdigest()
        identity["chat_template_sha256"] = template_hash
        tok.padding_side = "right"
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        if tok.pad_token_id is None:
            raise RuntimeError("Tokenizer has no padding/EOS ID")
        encoded = [encode_assistant_example(tok, row["messages"], cfg["max_length"], row["example_id"])
                   for row in rows]
        kept = [(row, enc) for row, enc in zip(rows, encoded) if not enc["dropped"]]
        singles = [(r, e) for r, e in kept if len(e["assistant_spans"]) == 1]
        multis = [(r, e) for r, e in kept if len(e["assistant_spans"]) > 1]
        if not singles or not multis:
            raise RuntimeError("Need at least one complete single-turn and multi-turn example within max_length")
        # Shortest complete example of each type limits smoke memory while covering both paths.
        selected = [min(singles, key=lambda pair: pair[1]["token_count"]),
                    min(multis, key=lambda pair: pair[1]["token_count"])]
        report = {"policy": "drop_whole_example; no token slicing or synthetic EOS", "max_length": cfg["max_length"],
                  "total": len(rows), "kept": len(kept), "dropped": sum(e["dropped"] for e in encoded),
                  "examples": [{"example_id": r["example_id"], "token_count": e["token_count"],
                                "dropped": e["dropped"], "reason": e["reason"]} for r, e in zip(rows, encoded)],
                  "selected_example_ids": [r["example_id"] for r, e in selected]}
        write_json("tokenization_report.json", report)
        for kind, (row, enc) in zip(["single", "multi"], selected):
            write_json(f"mask_inspection_{kind}.json", {"example_id": row["example_id"],
                       "assistant_spans": enc["assistant_spans"], "tokens": inspect_tokens(tok, enc)})
            print(kind, row["example_id"], "tokens", enc["token_count"], "assistant spans", enc["assistant_spans"])
        # Deliberate one-token overflow proves the policy preserves boundaries.
        row, enc = selected[0]
        overflow = encode_assistant_example(tok, row["messages"], enc["token_count"] - 1, row["example_id"])
        assert overflow["dropped"] and overflow["reason"] == "overflow"
        # Batch padding may use EOS as padding; labels are padded by position, never by token ID.
        padded = collate_examples([e for r, e in selected], tok.pad_token_id)
        for i, (r, enc) in enumerate(selected):
            assert padded["labels"][i][:enc["token_count"]] == enc["labels"]
            assert all(x == -100 for x in padded["labels"][i][enc["token_count"]:])
        manifest["checks"]["masks_and_truncation"] = True

        checkpoint("load_base")
        snapshot = snapshot_download(cfg["model_id"], revision=cfg["revision"],
                                     allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"], max_workers=2)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)

        def load_base():
            return AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                dtype=dtype, quantization_config=quant, device_map={"": 0},
                attn_implementation=cfg["attention_implementation"], trust_remote_code=False)

        def prepare_base(base, training):
            prefix = "training_" if training else "reload_"
            return prepare_frozen_quantized_base(base, dtype=dtype,
                device=torch.device("cuda", 0), prepare_kbit=prepare_model_for_kbit_training,
                training=training, stage_callback=lambda stage: memory(prefix + stage))

        torch.cuda.reset_peak_memory_stats(0)
        model = load_base()
        if not getattr(model, "is_loaded_in_4bit", False):
            raise RuntimeError("The base model was not loaded in 4-bit")
        memory("base_loaded")
        checkpoint("prepare_kbit_and_adapters")
        model = prepare_base(model, training=True)
        linears = [name for name, module in model.named_modules()
                   if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit))]
        targets = select_targets(linears, model.config.get_text_config(decoder=True).layer_types)
        adapter_cfg = LoraConfig(r=cfg["rank"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM",
            target_modules="^(?:" + "|".join(re.escape(n) for n in targets) + ")$")
        model = get_peft_model(model, adapter_cfg, revision=cfg["revision"])
        # Loading from a local snapshot must still produce a portable adapter.
        model.peft_config["default"].base_model_name_or_path = cfg["model_id"]
        named = dict(model.named_parameters())
        trainable = {n: p for n, p in named.items() if p.requires_grad}
        if not trainable or any(".lora_A." not in n and ".lora_B." not in n for n in trainable):
            raise RuntimeError("Only LoRA A/B parameters may be trainable")
        if any(any(part in n.lower() for part in ["visual", "vision", "lm_head", "embed_tokens"]) for n in trainable):
            raise RuntimeError("Vision/head/embedding trainability violation")
        # FP32 trainable tensors permit AMP GradScaler on T4; frozen compute stays FP16.
        for p in trainable.values():
            p.data = p.data.float()
        del p  # Do not retain a parameter reference after releasing the trained model.
        wrapped = sorted(n for n, module in model.named_modules() if hasattr(module, "lora_A"))
        expected_wrapped = sorted("base_model.model." + n for n in targets)
        if wrapped != expected_wrapped:
            raise RuntimeError("Attached adapters do not match the exact decoder target list")
        vision = [n for n in named if "vision" in n.lower() or "visual" in n.lower()]
        trainability = {"model_class": type(model.get_base_model()).__name__, "targets": targets,
                       "attached_modules": wrapped, "trainable_names": sorted(trainable),
                       "trainable_parameters": sum(p.numel() for p in trainable.values()),
                       "frozen_stored_elements": sum(p.numel() for p in named.values() if not p.requires_grad),
                       "preparation": "frozen embedding/head CPU staging only; all parameters restored to CUDA before computation",
                       "vision_parameters_present": len(vision),
                       "vision_status": "frozen" if vision else "absent: AutoModelForCausalLM loads text backbone only",
                       "frozen_embedding_head_dtype": str(dtype), "norm_dtype_policy": "PEFT FP32",
                       "preparation_staging": "Frozen input/output matrices on CPU for PEFT upcast; all parameters restored to GPU before compute",
                       "adapter_dtype": "torch.float32"}
        write_json("trainability.json", trainability)
        manifest["checks"]["frozen_base_and_exact_text_adapters"] = True
        memory("prepared_with_adapters")

        prompt_messages = [{"role": "system", "content": rows[0]["messages"][0]["content"]},
                           {"role": "user", "content": cfg["smoke_question"]}]
        prompt = tok.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_tensors="pt", return_dict=True).to("cuda:0")
        text_config = AutoConfig.from_pretrained(snapshot, local_files_only=True).get_text_config(decoder=True)
        generation = GenerationConfig.from_model_config(text_config)
        generation.max_new_tokens = cfg["smoke_max_new_tokens"]
        generation.do_sample, generation.num_beams = False, 1
        generation.temperature, generation.top_p, generation.top_k = 1.0, 1.0, 0
        generation.pad_token_id = tok.pad_token_id
        stop_policy = configure_probe_stop(tok, generation)
        write_json("smoke_prompt_and_decoding.json", {"messages": prompt_messages,
                   "input_ids": prompt["input_ids"][0].tolist(), "generation": generation.to_dict(),
                   "stop_policy": stop_policy,
                   "note": "Operational save/reload probe only; pilot quality comparison uses pilot_lock.json settings"})

        def generate_answer(current):
            current.eval()
            current.config.use_cache = True
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                output = current.generate(**prompt, generation_config=generation, logits_to_keep=1)
            ids = output[0, prompt["input_ids"].shape[1]:].tolist()
            text = tok.decode(ids, skip_special_tokens=True)
            if not ids or not text.strip():
                raise RuntimeError("Generation probe produced no answer")
            return {"token_ids": ids, "answer": text,
                    "finish_reason": "stop" if ids[-1] == generation.eos_token_id else "length"}

        checkpoint("baseline_probe")
        with model.disable_adapter():
            baseline = generate_answer(model)
        write_json("baseline_probe.json", baseline)
        model.config.use_cache = False
        model.train()
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=cfg["learning_rate"],
                                      weight_decay=cfg["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16),
                                     init_scale=cfg["amp_initial_scale"])
        initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        checkpoint("training")
        losses = []
        started = time.perf_counter()
        for step in range(cfg["steps"]):
            row, enc = selected[step % len(selected)]
            batch = {k: v.to("cuda:0") for k, v in
                     collate_examples([enc], tok.pad_token_id, as_tensors=True).items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                result = model(**batch, use_cache=False)
                loss = result.loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            missing = [n for n, p in trainable.items() if p.grad is None]
            nonfinite = [n for n, p in trainable.items() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            frozen_grads = [n for n, p in named.items() if not p.requires_grad and p.grad is not None]
            diagnostics = {"step": step + 1, "example_id": row["example_id"],
                           "loss": float(loss.detach()), "scaler_scale": scaler.get_scale(),
                           "missing": missing, "nonfinite": nonfinite, "frozen": frozen_grads}
            if missing or nonfinite or frozen_grads:
                write_json("gradient_diagnostics.json", {**diagnostics, "failure": "gradient_check"})
                raise RuntimeError(f"Gradient check failed: missing={missing}, nonfinite={nonfinite}, frozen={frozen_grads}")
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), cfg["max_grad_norm"], error_if_nonfinite=True)
            except RuntimeError:
                write_json("gradient_diagnostics.json", {**diagnostics, "failure": "gradient_norm_or_clip"})
                raise
            if float(grad_norm) <= 0:
                write_json("gradient_diagnostics.json", {**diagnostics, "failure": "all_zero_gradients",
                                                         "gradient_norm_before_clip": float(grad_norm)})
                raise RuntimeError("All adapter gradients are zero")
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps = {int(optimizer.state[p]["step"].item()) for p in trainable.values()}
            if optimizer_steps != {step + 1}:
                raise RuntimeError(f"Optimizer did not complete the expected update: {optimizer_steps}")
            metrics = {"step": step + 1, "example_id": row["example_id"], "loss": float(loss.detach()),
                       "verified_optimizer_step": step + 1,
                       "gradient_norm_before_clip": float(grad_norm), "finite_gradients": True,
                       "frozen_gradients_absent": True, "grad_scaler_scale": scaler.get_scale(),
                       "elapsed_seconds": time.perf_counter() - started}
            with (run_dir / "training_metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics, allow_nan=False) + "\n")
            losses.append(metrics["loss"])
            print(metrics, flush=True)
            del result, loss, batch
            memory(f"step_{step + 1}")
        changed = [n for n, p in trainable.items() if not torch.equal(initial[n], p.detach().cpu())]
        if not changed:
            raise RuntimeError("Optimizer did not change any adapter weights")
        manifest["checks"]["finite_loss_gradients_and_adapter_updates"] = True
        write_json("adapter_updates.json", {"changed_names": changed, "losses": losses})

        checkpoint("save_adapter")
        optimizer.zero_grad(set_to_none=True)
        trained_answer = generate_answer(model)
        write_json("trained_probe.json", trained_answer)
        adapter_dir = run_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
        tok.save_pretrained(adapter_dir)
        if not (adapter_dir / "adapter_config.json").is_file() or not (adapter_dir / "adapter_model.safetensors").is_file():
            raise RuntimeError("Adapter save incomplete")
        saved_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        if (saved_cfg.get("base_model_name_or_path") != cfg["model_id"]
                or saved_cfg.get("revision") != cfg["revision"]):
            raise RuntimeError("Saved adapter must identify the immutable base checkpoint")
        saved_state = {k: v.detach().cpu().clone() for k, v in
                       get_peft_model_state_dict(model, save_embedding_layers=False).items()}
        # Drop every parameter/optimizer reference before loading a fresh quantized base.
        del model, optimizer, trainable, named, initial, wrapped, scaler
        gc.collect()
        torch.cuda.empty_cache()
        memory("released_before_reload")
        checkpoint("fresh_base_reload")
        base = prepare_base(load_base(), training=False)
        reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        loaded_state = get_peft_model_state_dict(reloaded, save_embedding_layers=False)
        if saved_state.keys() != loaded_state.keys() or any(
                not torch.equal(v, loaded_state[k].detach().cpu().to(v.dtype)) for k, v in saved_state.items()):
            raise RuntimeError("Reloaded adapter tensors differ from saved state")
        if any(p.requires_grad for p in reloaded.parameters()):
            raise RuntimeError("Inference reload unexpectedly has trainable parameters")
        reloaded_answer = generate_answer(reloaded)
        write_json("reloaded_probe.json", reloaded_answer)
        if trained_answer["token_ids"] != reloaded_answer["token_ids"]:
            raise RuntimeError("Save/reload changed deterministic probe generation")
        memory("reloaded_and_generated")
        manifest["checks"]["fresh_reload_exact_adapter_tensors"] = True
        manifest["checks"]["fresh_reload_generation_matches"] = True
        manifest["status"] = "passed"
        checkpoint("complete")
        print("PASSED training compatibility preflight:", run_dir, flush=True)
        print(reloaded_answer["answer"], flush=True)
        return run_dir
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            memory("failure_" + manifest["stage"])
        except Exception:
            pass  # Preserve the original exception even if CUDA cannot report memory.
        write_json("run_manifest.json", manifest)
        print("FAILED at", manifest["stage"], "— reports preserved in", run_dir, flush=True)
        raise
