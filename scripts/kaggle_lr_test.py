"""Fresh 1e-5 adapter, fixed revised slice/order, 50 updates; Kaggle only."""
import hashlib
import json
from pathlib import Path
import re

def digest(payload):return hashlib.sha256(payload).hexdigest()
def fingerprint(value):return digest(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())

def run_lr_test(bundle_dir, output_root, settings=None, *, resume_checkpoint=None):
    import datetime
    import gc
    import importlib.metadata
    import os
    import platform
    import shutil
    import subprocess
    import sys
    import time

    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run model training only on Kaggle; local GPU work is disabled")
    from lr_test_protocol import CONDITIONS, build_order, validate_settings, validate_payloads, verify_metrics
    from build_training_pilot import validate_training_payload
    from training_masks import encode_assistant_example, inspect_tokens, collate_examples
    from kaggle_training import select_targets
    from kbit_preparation import prepare_frozen_quantized_base
    from pilot_checkpoints import save_checkpoint, load_checkpoint, verify_checkpoint
    bundle_dir, output_root = Path(bundle_dir), Path(output_root)
    pilot_bytes = (bundle_dir / "pilot_run_manifest.json").read_bytes()
    pilot_manifest = json.loads(pilot_bytes)
    protocol_pre = json.loads((bundle_dir / "protocol.json").read_bytes())
    if digest(pilot_bytes) != protocol_pre['pilot_run_manifest_sha256']:
        raise ValueError('Bundled original pilot manifest bytes differ from the frozen protocol')
    cfg = dict(protocol_pre["settings"] if settings is None else settings)
    validate_settings(cfg, pilot_manifest["identity"]["settings"])
    train_bytes = (bundle_dir / "train_slice.jsonl").read_bytes()
    eval_bytes = (bundle_dir / "model_inputs.jsonl").read_bytes()
    lock_bytes = (bundle_dir / "training_lock.json").read_bytes()
    protocol_bytes = (bundle_dir / "protocol.json").read_bytes()
    lock, protocol = json.loads(lock_bytes), json.loads(protocol_bytes)
    validate_payloads(train_bytes, eval_bytes, lock, protocol, pilot_manifest)
    rows = validate_training_payload(train_bytes, lock)
    eval_inputs = [json.loads(line) for line in eval_bytes.splitlines()]
    originals = list(Path('/kaggle/input').rglob('checkpoint-000050'))
    if len(originals) != 1:
        raise ValueError('Expected one saved revised lr1e4 step50 checkpoint before GPU work')
    original_dir = originals[0]
    original_cp = verify_checkpoint(original_dir, expected_identity=protocol_pre['control_identity'])
    if digest((original_dir/'checkpoint_manifest.json').read_bytes()) != protocol['original_checkpoint_manifest_sha256']:
        raise ValueError('Saved revised lr1e4 step50 checkpoint identity differs')
    control_bytes=(bundle_dir/'control_run_manifest.json').read_bytes()
    control=json.loads(control_bytes)
    if (digest(control_bytes)!=protocol_pre['control_run_manifest_sha256']
        or control['identity']!=protocol_pre['control_identity']
        or control['identity_sha256']!=protocol_pre['control_identity_sha256']
        or control['status']!='complete' or control['completed_steps']!=50
        or not control['checks'] or not all(v is True for v in control['checks'].values())):
        raise ValueError('Saved revised control manifest changed or incomplete')
    control_adapter=original_dir/'adapter/adapter_model.safetensors'
    if digest(control_adapter.read_bytes())!=protocol_pre['control_adapter_sha256']:
        raise ValueError('Saved revised comparator adapter differs')
    for helper in ['training_masks.py','kbit_preparation.py','kaggle_training.py','pilot_checkpoints.py','build_training_pilot.py']:
        if digest((bundle_dir/helper).read_bytes()) != pilot_manifest['identity']['helpers'][helper]:
            raise ValueError('Frozen training helper changed: '+helper)
    import torch
    import bitsandbytes as bnb
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    set_seed(cfg["seed"])
    # Preserve the passing T4 compute policy rather than silently switch dtype.
    dtype = torch.float16
    packages = {p: importlib.metadata.version(p) for p in
                ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface-hub", "tokenizers", "jinja2"]}
    if packages != protocol["packages"]:
        raise RuntimeError(f"Package stack differs from the passing smoke run: {packages}")
    gpu = {"name": torch.cuda.get_device_name(0),
           "compute_capability": list(torch.cuda.get_device_capability(0)),
           "total_bytes": torch.cuda.get_device_properties(0).total_memory}
    if gpu != pilot_manifest['identity']['gpu']:
        raise RuntimeError('The fixed comparison requires the original GPU-0 T4 runtime')
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], trust_remote_code=False)
    tok.padding_side = "right"
    if tok.pad_token_id != 248044 or tok.convert_tokens_to_ids("<|im_end|>") != 248046:
        raise RuntimeError("Tokenizer stop/pad IDs differ from the verified control")
    template_hash = digest(tok.chat_template.encode())
    if template_hash != protocol["chat_template_sha256"]:
        raise RuntimeError("Chat template differs from the frozen pilot protocol")
    encoded = [encode_assistant_example(tok, row["messages"], cfg["max_length"], row["example_id"])
               for row in rows]
    prompt_tokens = {case['case_id']: tok.apply_chat_template(case['messages'], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False) for case in eval_inputs}
    if any(not isinstance(ids, list) or not ids or len(ids) > 8192
           or any(type(token) is not int or token < 0 for token in ids)
           for ids in prompt_tokens.values()):
        raise ValueError('Invalid or over-budget evaluation tokenization; no truncation')
    order = build_order(encoded, cfg["seed"])
    order_ids = [rows[i]["example_id"] for i in order]
    if sum(e['supervised_tokens'] for e in encoded)!=4407:
        raise RuntimeError('Supervised token exposure changed')
    if fingerprint(order_ids) != protocol["training_order"]["ordered_example_ids_sha256"]:
        raise RuntimeError("Actual tokenized training order differs from the frozen protocol")
    if len(order) != cfg["steps"] or len(order) != 50:
        raise RuntimeError("This pilot must visit all 50 approved slice examples exactly once")
    identity = {"settings": cfg, "packages": packages,
                "model_id": cfg["model_id"], "revision": cfg["revision"],
                "dataset_sha256": digest(train_bytes), "pilot_lock_sha256": digest(lock_bytes),
                "protocol_sha256": digest(protocol_bytes), "evaluation_inputs_sha256": digest(eval_bytes),
                "chat_template_sha256": template_hash, "order_sha256": fingerprint(order_ids),
                "helpers": {path.name: digest(path.read_bytes()) for path in sorted(bundle_dir.glob("*.py"))},
                "gpu": gpu, "evaluation_prompt_token_ids_sha256": fingerprint(prompt_tokens),
                "dtype": str(dtype), "gpu_ids_used": [0], "python": platform.python_version(),
                "cuda_version": torch.version.cuda, "purpose": "controlled_lr_1e5_50_updates"}
    identity_sha = fingerprint(identity)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / f"pilot-{identity_sha[:12]}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "stage": "tokenization", "identity": identity,
                "identity_sha256": identity_sha, "checks": {}, "completed_steps": 0,
                "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None}

    def write_json(name, value):
        path = run_dir / name
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temp, path)

    def stage(name):
        manifest["stage"] = name
        write_json("run_manifest.json", manifest)

    def memory(name):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(0)
        data = {"stage": name, "allocated_bytes": torch.cuda.memory_allocated(0),
                "reserved_bytes": torch.cuda.memory_reserved(0), "free_bytes": free, "total_bytes": total,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        with (run_dir / "memory.jsonl").open("a") as stream:
            stream.write(json.dumps(data) + "\n")
        print("Memory:", data, flush=True)

    try:
        stage("tokenization")
        for name in ["training_lock.json", "protocol.json", "model_inputs.jsonl"]:
            shutil.copy2(bundle_dir / name, run_dir / name)
        (run_dir / "pip_freeze.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True))
        write_json("training_order.json", {"example_ids": order_ids, "sha256": fingerprint(order_ids),
                   "policy": "longest first, then seeded shuffle of remaining examples; each exactly once"})
        write_json("tokenization_report.json", {"total": len(rows), "retained": len(order), "dropped": 0,
                   "min_tokens": min(e["token_count"] for e in encoded),
                   "max_tokens": max(e["token_count"] for e in encoded),
                   "supervised_tokens": sum(e["supervised_tokens"] for e in encoded),
                   "examples": [{"example_id": e["example_id"], "token_count": e["token_count"],
                                 "supervised_tokens": e["supervised_tokens"]} for e in encoded]})
        inspections = [("longest", order[0])]
        for kind, count_test in [("single", lambda n: n == 1), ("multi", lambda n: n > 1)]:
            selected = next((i for i, e in enumerate(encoded) if count_test(len(e["assistant_spans"]))), None)
            if selected is not None:
                inspections.append((kind, selected))
        for name, index in inspections:
            write_json(f"mask_inspection_{name}.json", {"example_id": rows[index]["example_id"],
                       "spans": encoded[index]["assistant_spans"], "tokens": inspect_tokens(tok, encoded[index])})
        manifest["checks"]["all_50_masks_and_complete_answers"] = True

        stage("load_and_prepare_base")
        snapshot = snapshot_download(cfg["model_id"], revision=cfg["revision"],
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"], max_workers=2)
        if Path(snapshot).name != cfg["revision"]:
            raise RuntimeError("Base snapshot resolved to a different revision")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)

        def load_base(training):
            current = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                dtype=dtype, quantization_config=quant, device_map={"": 0},
                attn_implementation=cfg["attention_implementation"], trust_remote_code=False)
            if not getattr(current, "is_loaded_in_4bit", False):
                raise RuntimeError("Quantized base loading failed")
            return prepare_frozen_quantized_base(current, dtype=dtype, device=torch.device("cuda", 0),
                prepare_kbit=prepare_model_for_kbit_training, training=training,
                stage_callback=lambda name: memory(("training_" if training else "reload_") + name))

        torch.cuda.reset_peak_memory_stats(0)
        model = load_base(True)
        linears = [name for name, module in model.named_modules()
                   if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit))]
        targets = select_targets(linears, model.config.get_text_config(decoder=True).layer_types)
        adapter_cfg = LoraConfig(r=cfg["rank"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM",
            target_modules="^(?:" + "|".join(re.escape(n) for n in targets) + ")$")
        model = get_peft_model(model, adapter_cfg, revision=cfg["revision"])
        model.peft_config["default"].base_model_name_or_path = cfg["model_id"]
        named = dict(model.named_parameters())
        trainable = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
        if not trainable or any(".lora_A." not in n and ".lora_B." not in n for n in trainable):
            raise RuntimeError("Only LoRA A/B parameters may train")
        if any(any(part in n.lower() for part in ["vision", "visual", "lm_head", "embed_tokens"]) for n in trainable):
            raise RuntimeError("Trainable vision/head/embedding violation")
        for parameter in trainable.values():
            parameter.data = parameter.data.float()
        del parameter
        attached = sorted(name for name, module in model.named_modules() if hasattr(module, "lora_A"))
        if attached != sorted("base_model.model." + name for name in targets):
            raise RuntimeError("Adapter modules differ from the exact text projection target list")
        write_json("trainability.json", {"targets": targets, "attached_modules": attached,
                   "trainable_names": sorted(trainable), "trainable_parameters": sum(p.numel() for p in trainable.values()),
                   "frozen_stored_elements": sum(p.numel() for p in named.values() if not p.requires_grad),
                   "vision_parameters_present": sum("visual" in n.lower() or "vision" in n.lower() for n in named),
                   "adapter_dtype": "torch.float32", "base_embedding_head_dtype": str(dtype)})
        manifest["checks"]["exact_text_adapters_and_frozen_base"] = True
        memory("prepared_with_fresh_adapters")

        optimizer = torch.optim.AdamW(list(trainable.values()), lr=cfg["learning_rate"],
                                      weight_decay=cfg["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=cfg["amp_initial_scale"])
        metrics_path = run_dir / "training_metrics.jsonl"
        start_index = 0
        if resume_checkpoint is not None:
            stage("restore_checkpoint")
            source = Path(resume_checkpoint)
            start_index = load_checkpoint(source, identity, order_ids, model, optimizer, scaler, torch)
            if not 0 <= start_index <= len(order):
                raise ValueError("Invalid checkpoint data position")
            shutil.copy2(source / "training_metrics.jsonl", metrics_path)
            manifest["checks"]["resumable_checkpoint_round_trip"] = True
            source_run = source.parent.parent
            if (source_run / "run_manifest.json").is_file():
                previous = json.loads((source_run / "run_manifest.json").read_text())
                if previous.get("identity_sha256") != identity_sha:
                    raise ValueError("Evaluation resume source has a different identity")
                for name in [f"{condition}_responses.jsonl" for condition in CONDITIONS]:
                    if (source_run / name).exists():
                        shutil.copy2(source_run / name, run_dir / name)
        else:
            metrics_path.write_text("")
        manifest["completed_steps"] = start_index
        initial = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
        model.train()
        model.config.use_cache = False
        stage("training")
        if start_index == 0:
            zero = save_checkpoint(run_dir / "checkpoints", 0, identity, order_ids, model,
                                   optimizer, scaler, torch, metrics_path=metrics_path)
            zero_manifest=json.loads((zero/'checkpoint_manifest.json').read_bytes())
            if zero_manifest['files']['adapter/adapter_model.safetensors']!=protocol['control_initial_adapter_sha256']:
                raise ValueError('Fresh adapter initialization differs from the lr1e4 control')
            manifest['checks']['matched_initial_adapter_bytes'] = True
            print("Recovery checkpoint:", zero, flush=True)
        session_started = time.perf_counter()
        for index in range(start_index, len(order)):
            selected = order[index]
            row, enc = rows[selected], encoded[selected]
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to("cuda:0") for key, value in
                     collate_examples([enc], tok.pad_token_id, as_tensors=True).items()}
            with torch.autocast("cuda", dtype=dtype):
                result = model(**batch, use_cache=False)
                loss = result.loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at example {row['example_id']}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            missing = [name for name, p in trainable.items() if p.grad is None]
            nonfinite = [name for name, p in trainable.items() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            frozen = [name for name, p in named.items() if not p.requires_grad and p.grad is not None]
            if missing or nonfinite or frozen:
                write_json("gradient_failure.json", {"step": index + 1, "example_id": row["example_id"],
                           "loss": float(loss.detach()), "scale": scaler.get_scale(),
                           "missing": missing, "nonfinite": nonfinite, "frozen": frozen})
                raise FloatingPointError(f"Gradient checks failed at step {index + 1}")
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), cfg["max_grad_norm"], error_if_nonfinite=True)
            if float(norm) <= 0:
                raise RuntimeError(f"All adapter gradients are zero at step {index + 1}")
            scaler.step(optimizer)
            scaler.update()
            verified = {int(optimizer.state[p]["step"].item()) for p in trainable.values()}
            if verified != {index + 1}:
                raise RuntimeError("Optimizer update was skipped or checkpoint counters differ")
            metric = {"step": index + 1, "verified_optimizer_step": index + 1,
                      "example_id": row["example_id"], "input_tokens": enc["token_count"],
                      "supervised_tokens": enc["supervised_tokens"], "loss": float(loss.detach()),
                      "gradient_norm_before_clip": float(norm), "grad_scaler_scale": scaler.get_scale(),
                      "finite_gradients": True, "frozen_gradients_absent": True,
                      "session_elapsed_seconds": time.perf_counter() - session_started}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(metric, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            manifest["completed_steps"] = index + 1
            if index == 0 or (index + 1) % cfg["logging_steps"] == 0:
                print("Training:", metric, flush=True)
            del result, loss, batch
            if index == 0:
                manifest["checks"]["longest_example_finite_and_fits"] = True
                memory("longest_example_step_1")
            if (index + 1) % cfg["save_steps"] == 0 or index + 1 == len(order):
                saved = save_checkpoint(run_dir / "checkpoints", index + 1, identity, order_ids,
                    model, optimizer, scaler, torch, metrics_path=metrics_path)
                if index + 1 == cfg["save_steps"]:
                    restored_index = load_checkpoint(saved, identity, order_ids, model, optimizer, scaler, torch)
                    if restored_index != index + 1:
                        raise RuntimeError("Checkpoint round-trip changed the next data index")
                    manifest["checks"]["resumable_checkpoint_round_trip"] = True
                    print("Verified adapter/optimizer/scaler/RNG restore at step", restored_index, flush=True)
                manifest["latest_checkpoint"] = str(saved)
                stage("training")
                memory(f"checkpoint_{index + 1}")
                print(f"CHECKPOINT {index + 1}/{len(order)}:", saved, flush=True)

        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if (len(metrics) != 50 or [r["example_id"] for r in metrics] != order_ids
                or [r["verified_optimizer_step"] for r in metrics] != list(range(1, 51))):
            raise RuntimeError("Training did not visit every reviewed example exactly once")
        verify_metrics(metrics, order_ids)
        changed = [name for name, parameter in trainable.items()
                   if not torch.equal(initial[name], parameter.detach().cpu())]
        if start_index < 50 and not changed:
            raise RuntimeError("No adapter weights changed during this session")
        manifest["checks"]["all_50_finite_verified_optimizer_steps"] = True
        manifest["checks"]["all_examples_seen_exactly_once"] = True
        # A restored complete checkpoint already carries the first-step evidence.
        manifest["checks"]["longest_example_finite_and_fits"] = metrics[0]["input_tokens"] == max(e["token_count"] for e in encoded)
        write_json("training_summary.json", {"completed_optimizer_steps": len(metrics),
                   "unique_training_examples": len({r["example_id"] for r in metrics}),
                   "mean_training_loss": sum(r["loss"] for r in metrics) / len(metrics),
                   "session_changed_adapter_tensors": changed,
                   "first_25_mean_loss": sum(r["loss"] for r in metrics[:25]) / 25,
                   "last_25_mean_loss": sum(r["loss"] for r in metrics[-25:]) / 25,
                   "quality_improvement_established": False})

        stage("final_adapter_and_reload")
        optimizer.zero_grad(set_to_none=True)
        final_dir = run_dir / "final_adapter"
        model.save_pretrained(final_dir, safe_serialization=True, save_embedding_layers=False)
        tok.save_pretrained(final_dir)
        saved_state = {key: value.detach().cpu().clone() for key, value in
                       get_peft_model_state_dict(model, save_embedding_layers=False).items()}
        saved_cfg = json.loads((final_dir / "adapter_config.json").read_text())
        if saved_cfg.get("base_model_name_or_path") != cfg["model_id"] or saved_cfg.get("revision") != cfg["revision"]:
            raise RuntimeError("Final adapter does not identify the pinned base")
        generation = GenerationConfig.from_model_config(model.config.get_text_config(decoder=True))
        generation.do_sample, generation.num_beams = False, 1
        generation.temperature, generation.top_p, generation.top_k = 1.0, 1.0, 0
        generation.pad_token_id = tok.pad_token_id
        generation.use_cache = True
        generation.eos_token_id = tok.convert_tokens_to_ids("<|im_end|>")
        if generation.eos_token_id != cfg["evaluation_eos_token_id"]:
            raise RuntimeError("Evaluation stop token differs from the frozen protocol")
        generation.max_new_tokens = cfg["smoke_max_new_tokens"]
        prompt_messages = [rows[0]["messages"][0], {"role": "user", "content": cfg["smoke_question"]}]

        def generate(current, messages, config):
            current.eval()
            current.config.use_cache = True
            inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt", return_dict=True).to("cuda:0")
            if inputs["input_ids"].shape[1] > 8192:
                raise ValueError("Evaluation prompt exceeds 8192 tokens; no silent truncation")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                output = current.generate(**inputs, generation_config=config, logits_to_keep=1)
            torch.cuda.synchronize()
            ids = output[0, inputs["input_ids"].shape[1]:].tolist()
            if not ids:
                raise RuntimeError("Generation returned no tokens")
            return {"token_ids": ids, "token_count": len(ids), "response": tok.decode(ids, skip_special_tokens=True),
                    "finish_reason": "stop" if ids[-1] == config.eos_token_id else "length",
                    "generation_seconds": time.perf_counter() - started,
                    "input_sha256": fingerprint(inputs["input_ids"][0].tolist())}

        trained_probe = generate(model, prompt_messages, generation)
        write_json("trained_probe.json", trained_probe)
        del model, optimizer, scaler, trainable, named, initial
        gc.collect()
        torch.cuda.empty_cache()
        memory("released_before_fresh_reload")
        base = load_base(False)
        reloaded = PeftModel.from_pretrained(base, final_dir, is_trainable=False)
        loaded_state = get_peft_model_state_dict(reloaded, save_embedding_layers=False)
        if saved_state.keys() != loaded_state.keys() or any(not torch.equal(value, loaded_state[key].detach().cpu().to(value.dtype))
                                                         for key, value in saved_state.items()):
            raise RuntimeError("Freshly reloaded adapter tensors differ")
        if any(p.requires_grad for p in reloaded.parameters()):
            raise RuntimeError("Inference reload has trainable parameters")
        reloaded_probe = generate(reloaded, prompt_messages, generation)
        write_json("reloaded_probe.json", reloaded_probe)
        if trained_probe["token_ids"] != reloaded_probe["token_ids"]:
            raise RuntimeError("Fresh reload changed greedy probe token IDs")
        manifest["checks"]["fresh_reload_exact_adapter_tensors"] = True
        manifest["checks"]["fresh_reload_generation_matches"] = True
        memory("fresh_adapter_reload_verified")
        del saved_state, loaded_state

        if cfg["run_evaluation"]:
            stage("matched_fresh_and_regression_evaluation")
            del reloaded, base
            gc.collect()
            torch.cuda.empty_cache()
            generation.max_new_tokens = cfg["evaluation_max_new_tokens"]
            gen_dict = generation.to_dict()
            if any(gen_dict.get(key) != value for key, value in protocol["evaluation"]["generation"].items()):
                raise RuntimeError("Evaluation decoding differs from frozen v2 protocol")
            from kaggle_checkpoint_compare import read_evaluation_journal, atomic_journal
            from pilot_checkpoints import verify_checkpoint
            from safetensors.torch import load_file
            candidates = list(Path('/kaggle/input').rglob('checkpoint-000050'))
            if len(candidates) != 1:
                raise ValueError("Expected exactly one saved revised lr1e4 step50 checkpoint attachment")
            original_dir = candidates[0]
            original_cp = verify_checkpoint(original_dir, expected_identity=protocol['control_identity'])
            if digest((original_dir/'checkpoint_manifest.json').read_bytes()) != protocol['original_checkpoint_manifest_sha256']:
                raise ValueError("Saved revised lr1e4 step50 checkpoint manifest differs")
            write_json("comparison_manifest.json", {"identity_sha256": identity_sha,
                "conditions": CONDITIONS, "generation": gen_dict, "generation_sha256": fingerprint(gen_dict),
                "enable_thinking": False, "evaluation_inputs_sha256": digest(eval_bytes),
                "evaluation_protocol_sha256": protocol['evaluation']['source_protocol_sha256'],
                "prompt_policy": "Identical frozen v3 fresh and v2 regression messages, no reference passages",
                "fresh_base_before_any_evaluation_adapter": True,
                "original_checkpoint_manifest_sha256": digest((original_dir/'checkpoint_manifest.json').read_bytes())})
            all_results = {}
            eval_model = load_base(False)
            if getattr(eval_model, 'peft_config', None): raise RuntimeError("Untouched evaluation base has an adapter")
            manifest['checks']['evaluation_base_untouched'] = True
            for condition in CONDITIONS:
                if condition == 'revised_lr1e4_step50':
                    eval_model = PeftModel.from_pretrained(eval_model, original_dir/'adapter', is_trainable=False)
                    saved = load_file(str(original_dir/'adapter/adapter_model.safetensors'),device='cpu')
                    actual = get_peft_model_state_dict(eval_model,save_embedding_layers=False)
                    if saved.keys()!=actual.keys() or any(not torch.equal(v,actual[k].detach().cpu().to(v.dtype)) for k,v in saved.items()):
                        raise RuntimeError("Saved revised lr1e4 step50 tensor reload differs")
                    del saved,actual
                    manifest['checks']['original_step50_exact_tensors'] = True
                elif condition == 'revised_lr1e5_step50':
                    del eval_model
                    gc.collect()
                    torch.cuda.empty_cache()
                    eval_model = PeftModel.from_pretrained(load_base(False), final_dir, is_trainable=False)
                    saved = load_file(str(final_dir/'adapter_model.safetensors'), device='cpu')
                    actual = get_peft_model_state_dict(eval_model, save_embedding_layers=False)
                    if saved.keys() != actual.keys() or any(not torch.equal(v, actual[k].detach().cpu().to(v.dtype)) for k, v in saved.items()):
                        raise RuntimeError('Revised step50 evaluation tensor reload differs')
                    del saved, actual
                    manifest['checks']['revised_eval_exact_tensors'] = True
                    manifest['checks']['revised_eval_fresh_base_reload'] = True
                if any(p.requires_grad for p in eval_model.parameters()): raise RuntimeError("Evaluation parameters remain trainable")
                if any(str(p.device) != 'cuda:0' for p in eval_model.parameters()): raise RuntimeError('Evaluation parameters escaped GPU 0')
                path = run_dir/f'{condition}_responses.jsonl'
                records = read_evaluation_journal(path,identity_sha,condition,eval_inputs,gen_dict,prompt_tokens)
                for case in eval_inputs[len(records):]:
                    set_seed(cfg['seed'])
                    generated = generate(eval_model,case['messages'],generation)
                    expected_tokens = prompt_tokens[case['case_id']]
                    if generated['input_sha256']!=fingerprint(expected_tokens):raise RuntimeError("Input tokens changed")
                    row = {'identity_sha256':identity_sha,'condition':condition,'case_id':case['case_id'],
                        'category':case['category'],'split':case['split'],'prompt_sha256':fingerprint(case['messages']),
                        'generation_sha256':fingerprint(gen_dict),'input_token_ids':expected_tokens,**generated}
                    atomic_journal(path,records+[row]);records.append(row)
                    print('Evaluation:',condition,f'{len(records)}/{len(eval_inputs)}',case['case_id'],row['finish_reason'],flush=True)
                all_results[condition] = read_evaluation_journal(path,identity_sha,condition,eval_inputs,gen_dict,prompt_tokens)
            summary = {'complete':all(len(rs)==len(eval_inputs) for rs in all_results.values()),
                'expected_per_condition':len(eval_inputs),'completed_per_condition':{c:len(rs) for c,rs in all_results.items()},
                'truncations_by_condition':{c:sum(r['finish_reason']=='length' for r in rs) for c,rs in all_results.items()},
                'quality_scores_assigned':False}
            write_json('evaluation_summary.json',summary)
            if not summary['complete']:raise RuntimeError("Incomplete v2 evaluation")
            manifest['checks']['matched_fresh_and_regression_evaluation_complete'] = True
            del eval_model
            gc.collect()
            torch.cuda.empty_cache()
        if not manifest["checks"] or not all(manifest["checks"].values()):
            raise RuntimeError("Pilot checks were not all satisfied")
        manifest["status"] = "complete"
        stage("complete")
        memory("complete")
        # A compact download preserves the final adapter, inputs and reports.
        # Full resumable checkpoints remain in the saved Kaggle output tree.
        import zipfile
        with zipfile.ZipFile(run_dir / "lr_test_delivery.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(run_dir.iterdir()):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".txt"}:
                    archive.write(path, path.name)
            for path in sorted(final_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(run_dir))
            for path in sorted(bundle_dir.iterdir()):
                if path.is_file():
                    archive.write(path, "inputs/" + path.name)
        print("DATA REVISION PILOT COMPLETE:", run_dir, flush=True)
        return run_dir
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            memory("failure_" + manifest["stage"])
        except Exception:
            pass
        write_json("run_manifest.json", manifest)
        print("PILOT FAILED at", manifest["stage"], "completed steps", manifest["completed_steps"],
              "reports and recovery checkpoints:", run_dir, flush=True)
        raise
