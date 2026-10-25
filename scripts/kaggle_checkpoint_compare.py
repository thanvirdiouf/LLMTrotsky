"""Inference-only fresh-base / step-50 / step-500 diagnostic on Kaggle.

The decoding and response-journal fields follow the verified pilot generation
path. No optimizer, training loop or checkpoint training-state deserialization
is used. Importing this module requires only the standard library.
"""
import hashlib
import json
import os
from pathlib import Path


CONDITIONS = ("fresh_base", "step_50", "step_500")
MODEL = {"model_id": "Qwen/Qwen3.5-9B",
         "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
GENERATION = {"do_sample": False, "eos_token_id": 248046,
              "max_new_tokens": 512, "num_beams": 1,
              "pad_token_id": 248044, "use_cache": True}
KERNEL_SOURCE = "thanvirdiouf/trotsky-qlora-500-example-pilot/2"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode())


def atomic_bytes(path, payload):
    """Replace the complete file, never leave a partly appended JSONL row."""
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False,
                                  allow_nan=False) + "\n").encode())


def atomic_journal(path, rows):
    atomic_bytes(path, b"".join((json.dumps(row, ensure_ascii=False,
                                           allow_nan=False) + "\n").encode()
                               for row in rows))


def validate_payloads(input_bytes, protocol, pilot_manifest):
    """Check the frozen runtime projection and reject reference-bearing inputs."""
    if protocol.get("model") != MODEL or protocol.get("generation") != GENERATION:
        raise ValueError("Model revision or primary decoding differs from the frozen comparison")
    if (protocol.get("conditions") != list(CONDITIONS)
            or protocol.get("cases") != 24 or protocol.get("seed") != 42
            or protocol.get("enable_thinking") is not False
            or protocol.get("max_input_tokens") != 8192
            or protocol.get("pilot_kernel_source") != KERNEL_SOURCE):
        raise ValueError("Comparison conditions/input policy differ from the frozen primary run")
    if digest(input_bytes) != protocol["files"]["model_inputs.jsonl"]:
        raise ValueError("Fresh model-input checksum mismatch")
    identity = pilot_manifest["identity"]
    if (pilot_manifest.get("status") != "complete"
            or pilot_manifest.get("completed_steps") != 500
            or pilot_manifest.get("identity_sha256") != fingerprint(identity)
            or protocol.get("pilot_identity_sha256") != fingerprint(identity)
            or protocol.get("pilot_identity") != identity
            or identity.get("packages") != protocol.get("packages")
            or identity.get("model_id") != MODEL["model_id"]
            or identity.get("revision") != MODEL["revision"]
            or identity.get("chat_template_sha256") != protocol.get("chat_template_sha256")):
        raise ValueError("Archived pilot identity does not match the declared comparison")
    settings = identity["settings"]
    if (settings.get("quantization") != "nf4_double"
            or settings.get("attention_implementation") != "sdpa"
            or identity.get("dtype") != "torch.float16"
            or identity.get("gpu_ids_used") != [0]):
        raise ValueError("Pilot quantization/precision/device policy changed")
    rows = [json.loads(line) for line in input_bytes.splitlines()]
    seen = set()
    for row in rows:
        if (not isinstance(row, dict)
                or set(row) != {"case_id", "category", "split", "messages"}
                or not isinstance(row["case_id"], str) or not row["case_id"]
                or row["case_id"] in seen or not isinstance(row["category"], str)
                or row["split"] != "fresh_diagnostic_holdout"):
            raise ValueError("Only unique fresh model-input-only records are allowed")
        messages = row["messages"]
        if (not isinstance(messages, list) or len(messages) < 2
                or messages[0].get("role") != "system"
                or messages[-1].get("role") != "user"):
            raise ValueError("Invalid evaluation message sequence")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user"}
                    or not isinstance(message["content"], str) or not message["content"]):
                raise ValueError("Model inputs may contain only system/user text messages")
        seen.add(row["case_id"])
    if len(rows) != protocol["cases"]:
        raise ValueError("Expected all 24 frozen cases")
    return rows


def find_checkpoints(input_root, expected_identity, verify):
    """Require the two declared complete pilot checkpoints; never select by score."""
    result = {}
    for step in (50, 500):
        candidates = sorted(Path(input_root).rglob(f"checkpoint-{step:06d}"))
        matches = []
        for path in candidates:
            # A wrong identity is an error, rather than permission to select a
            # different checkpoint from an unrelated attachment silently.
            manifest = verify(path, expected_identity=expected_identity)
            if manifest["completed_steps"] != step:
                raise ValueError("Checkpoint step does not match the declared condition")
            matches.append((path, manifest))
        if len(matches) != 1:
            raise ValueError(f"Expected one complete pilot checkpoint at step {step}; found {len(matches)}")
        result[f"step_{step}"] = matches[0]
    if result["step_50"][1]["order_ids"] != result["step_500"][1]["order_ids"]:
        raise ValueError("Pilot checkpoints do not share their frozen example order")
    return result


def tokenize_inputs(tokenizer, inputs, max_input_tokens):
    """Ask explicitly for a token list; newer tokenizers default to a mapping."""
    prompts = {}
    for case in inputs:
        value = tokenizer.apply_chat_template(case["messages"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False,
                    return_dict=False)
        if (not isinstance(value, list) or not value
                or any(type(token) is not int or token < 0 for token in value)):
            raise ValueError("Chat template must return a nonempty integer token list")
        if len(value) > max_input_tokens:
            raise ValueError("Evaluation input exceeds the frozen token budget; no truncation")
        prompts[case["case_id"]] = value
    return prompts


def read_evaluation_journal(path, identity_sha256, condition, inputs, generation,
                            prompt_tokens):
    """Pilot journal validation plus exact input-token and EOS checks."""
    path = Path(path)
    if not path.exists():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError("Incomplete response journal; refusing implicit recovery")
    allowed = {case["case_id"]: case for case in inputs}
    seen, rows = set(), []
    for line in payload.splitlines():
        row = json.loads(line)
        key = row.get("case_id")
        if (key not in allowed or key in seen or row.get("condition") != condition
                or row.get("identity_sha256") != identity_sha256
                or row.get("prompt_sha256") != fingerprint(allowed[key]["messages"])
                or row.get("generation_sha256") != fingerprint(generation)
                or row.get("input_token_ids") != prompt_tokens[key]
                or row.get("input_sha256") != fingerprint(prompt_tokens[key])
                or not isinstance(row.get("response"), str)
                or not isinstance(row.get("token_ids"), list)
                or any(type(token) is not int or token < 0 for token in row["token_ids"])
                or row.get("finish_reason") not in {"stop", "length"}):
            raise ValueError(f"Invalid/mixed evaluation journal row: {key}")
        tokens = row["token_ids"]
        if (row.get("token_count") != len(tokens) or not tokens
                or len(tokens) > generation["max_new_tokens"]
                or row["finish_reason"] != ("stop" if tokens[-1] == generation["eos_token_id"] else "length")
                or (row["finish_reason"] == "length" and len(tokens) != generation["max_new_tokens"])):
            raise ValueError("Evaluation token count/finish reason differs from decoding")
        seen.add(key)
        rows.append(row)
    # Atomic writes always retain an ordered prefix. Reject hand-combined rows.
    if [r["case_id"] for r in rows] != [c["case_id"] for c in inputs[:len(rows)]]:
        raise ValueError("Evaluation journal is not an ordered prefix of the frozen cases")
    return rows


def export_results(run_dir, all_results, inputs, identity_sha):
    """Export completeness/truncation evidence and blank review cells only."""
    import csv
    import io
    expected = [case["case_id"] for case in inputs]
    summary = {"identity_sha256": identity_sha, "expected_per_condition": len(inputs),
               "completed_per_condition": {k: len(all_results[k]) for k in CONDITIONS},
               "truncations_by_condition": {k: sum(row["finish_reason"] == "length" for row in all_results[k]) for k in CONDITIONS},
               "complete": all([row["case_id"] for row in all_results[k]] == expected for k in CONDITIONS),
               "quality_scores_assigned": False}
    atomic_json(Path(run_dir) / "evaluation_summary.json", summary)
    by_condition = {kind: {r["case_id"]: r for r in values} for kind, values in all_results.items()}
    fields = ["case_id", "category", "split", "messages"]
    for kind in CONDITIONS:
        fields += [kind + "_response", kind + "_finish_reason", kind + "_token_count",
                   kind + "_critical_failure", kind + "_chronology", kind + "_correction",
                   kind + "_evidence_calibration", kind + "_answer_detail", kind + "_review_notes"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for case in inputs:
        item = {k: case[k] for k in ("case_id", "category", "split")}
        item["messages"] = json.dumps(case["messages"], ensure_ascii=False)
        for kind in CONDITIONS:
            response = by_condition[kind].get(case["case_id"], {})
            item[kind + "_response"] = response.get("response", "")
            item[kind + "_finish_reason"] = response.get("finish_reason", "")
            item[kind + "_token_count"] = response.get("token_count", "")
        writer.writerow(item)
    atomic_bytes(Path(run_dir) / "paired_answers.csv", stream.getvalue().encode())
    return summary


def run_comparison(bundle_dir, output_root, *, input_root="/kaggle/input", resume_dir=None):
    import datetime
    import fcntl
    import gc
    import importlib.metadata
    import platform
    import subprocess
    import time
    import zipfile

    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run model inference only on Kaggle; local GPU work is disabled")
    bundle_dir, output_root = Path(bundle_dir), Path(output_root)
    input_bytes = (bundle_dir / "model_inputs.jsonl").read_bytes()
    protocol_bytes = (bundle_dir / "runtime_protocol.json").read_bytes()
    pilot_bytes = (bundle_dir / "pilot_run_manifest.json").read_bytes()
    protocol, pilot_manifest = json.loads(protocol_bytes), json.loads(pilot_bytes)
    if digest(pilot_bytes) != protocol["pilot_run_manifest_sha256"]:
        raise ValueError("Archived pilot manifest checksum mismatch")
    inputs = validate_payloads(input_bytes, protocol, pilot_manifest)
    from pilot_checkpoints import verify_checkpoint
    from kbit_preparation import prepare_frozen_quantized_base
    checkpoints = find_checkpoints(input_root, pilot_manifest["identity"], verify_checkpoint)
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel, prepare_model_for_kbit_training
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed

    packages = {key: importlib.metadata.version(key) for key in protocol["packages"]}
    if packages != protocol["packages"]:
        raise RuntimeError(f"Package stack differs from the frozen pilot: {packages}")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle T4 GPU accelerator")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    gpu = {"name": torch.cuda.get_device_name(0),
           "compute_capability": list(torch.cuda.get_device_capability(0)),
           "total_bytes": torch.cuda.get_device_properties(0).total_memory}
    if gpu != pilot_manifest["identity"]["gpu"]:
        raise RuntimeError("Primary comparison requires the pilot's GPU-0 T4 runtime")
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(**{"pretrained_model_name_or_path": MODEL["model_id"],
                                                   "revision": MODEL["revision"], "trust_remote_code": False})
    tokenizer.padding_side = "right"
    template_hash = digest(tokenizer.chat_template.encode())
    if template_hash != protocol["chat_template_sha256"]:
        raise RuntimeError("Tokenizer chat template differs from the frozen pilot")
    if (tokenizer.convert_tokens_to_ids("<|im_end|>") != GENERATION["eos_token_id"]
            or tokenizer.pad_token_id != GENERATION["pad_token_id"]):
        raise RuntimeError("Tokenizer stop/pad identity changed")
    prompt_tokens = tokenize_inputs(tokenizer, inputs, protocol["max_input_tokens"])
    for helper in ("kbit_preparation.py", "training_masks.py", "pilot_checkpoints.py"):
        if digest((bundle_dir / helper).read_bytes()) != pilot_manifest["identity"]["helpers"][helper]:
            raise ValueError("Frozen pilot helper changed: " + helper)
    snapshot = snapshot_download(MODEL["model_id"], revision=MODEL["revision"],
                allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"], max_workers=2)
    if Path(snapshot).name != MODEL["revision"]:
        raise RuntimeError("Model snapshot did not resolve to the exact pinned revision")
    quantization = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True, "compute_dtype": "torch.float16"}
    # GenerationConfig.from_model_config and every explicit override match the
    # tested pilot; construct before loading adapter state.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    generation = GenerationConfig.from_model_config(config.get_text_config(decoder=True))
    generation.do_sample, generation.num_beams = False, 1
    generation.temperature, generation.top_p, generation.top_k = 1.0, 1.0, 0
    generation.pad_token_id = tokenizer.pad_token_id
    generation.use_cache, generation.eos_token_id = True, GENERATION["eos_token_id"]
    generation.max_new_tokens = GENERATION["max_new_tokens"]
    gen_dict = generation.to_dict()
    if any(gen_dict.get(k) != value for k, value in GENERATION.items()):
        raise RuntimeError("Actual generation settings differ from the frozen primary run")
    identity = {"model": MODEL, "packages": packages, "python": platform.python_version(),
                "cuda_version": torch.version.cuda, "gpu": gpu, "gpu_ids_used": [0],
                "dtype": str(dtype), "quantization": quantization,
                "attention_implementation": "sdpa", "kbit_preparation_training": False,
                "seed": 42, "enable_thinking": False, "conditions": list(CONDITIONS),
                "source_protocol_sha256": protocol["source_protocol_sha256"],
                "runtime_protocol_sha256": digest(protocol_bytes),
                "model_inputs_sha256": digest(input_bytes), "pilot_run_manifest_sha256": digest(pilot_bytes),
                "pilot_identity_sha256": protocol["pilot_identity_sha256"],
                "chat_template_sha256": template_hash, "generation": gen_dict,
                "generation_sha256": fingerprint(gen_dict), "prompt_token_ids_sha256": fingerprint(prompt_tokens),
                "helpers": {p.name: digest(p.read_bytes()) for p in sorted(bundle_dir.glob("*.py"))},
                "checkpoints": {kind: {"step": manifest["completed_steps"],
                    "manifest_sha256": digest((path / "checkpoint_manifest.json").read_bytes()),
                    "adapter_sha256": manifest["files"]["adapter/adapter_model.safetensors"],
                    "identity_sha256": manifest["identity_sha256"]}
                    for kind, (path, manifest) in checkpoints.items()},
                "purpose": "fresh_holdout_v1_inference_only"}
    identity_sha = fingerprint(identity)
    if resume_dir is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = output_root / f"compare-{identity_sha[:12]}-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = Path(resume_dir)
        previous = json.loads((run_dir / "run_manifest.json").read_bytes())
        if previous.get("identity") != identity or previous.get("identity_sha256") != identity_sha:
            raise ValueError("Resume identity differs: refuse mixed packages/helpers/inputs/checkpoints")
    # Advisory lock prevents two cells from concurrently replacing journals.
    lock = (run_dir / ".run.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {"status": "running", "stage": "preflight", "identity": identity,
                "identity_sha256": identity_sha, "checks": {}, "quality_scores_assigned": False}
    all_results = {kind: read_evaluation_journal(run_dir / f"{kind}_responses.jsonl", identity_sha,
                   kind, inputs, gen_dict, prompt_tokens) for kind in CONDITIONS}

    def stage(name):
        manifest["stage"] = name
        atomic_json(run_dir / "run_manifest.json", manifest)
        print("Stage:", name, flush=True)

    def memory(name):
        with (run_dir / "memory.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": name, "allocated_bytes": torch.cuda.memory_allocated(0),
                "reserved_bytes": torch.cuda.memory_reserved(0),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def load_base():
        set_seed(42)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        current = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                  dtype=dtype, quantization_config=quant, device_map={"": 0},
                  attn_implementation="sdpa", trust_remote_code=False)
        if not getattr(current, "is_loaded_in_4bit", False):
            raise RuntimeError("NF4 base loading failed")
        current = prepare_frozen_quantized_base(current, dtype=dtype, device=torch.device("cuda", 0),
                  prepare_kbit=prepare_model_for_kbit_training, training=False,
                  stage_callback=memory)
        if getattr(current, "peft_config", None) or any(p.requires_grad for p in current.parameters()):
            raise RuntimeError("Fresh base must be untouched by adapters and frozen for inference")
        if any("visual" in name for name, _ in current.named_modules()):
            raise RuntimeError("Expected pilot text-only architecture without a vision tower")
        return current

    def attach(current, condition):
        path, checkpoint = checkpoints[condition]
        adapter_dir = path / "adapter"
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_bytes())
        settings = pilot_manifest["identity"]["settings"]
        if (adapter_config.get("base_model_name_or_path") != MODEL["model_id"]
                or adapter_config.get("revision") != MODEL["revision"]
                or adapter_config.get("r") != settings["rank"]
                or adapter_config.get("lora_alpha") != settings["lora_alpha"]
                or adapter_config.get("lora_dropout") != settings["lora_dropout"]):
            raise RuntimeError("Adapter base identity or LoRA settings differ from the frozen pilot")
        current = PeftModel.from_pretrained(current, adapter_dir, is_trainable=False)
        saved = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
        actual = get_peft_model_state_dict(current, save_embedding_layers=False)
        if set(saved) != set(actual) or any(not torch.equal(value, actual[key].detach().cpu().to(value.dtype)) for key, value in saved.items()):
            raise RuntimeError("Loaded checkpoint adapter tensors differ")
        if any(p.requires_grad for p in current.parameters()):
            raise RuntimeError("Inference adapter has trainable parameters")
        if any(str(p.device) != "cuda:0" for p in current.parameters()):
            raise RuntimeError("Inference parameters escaped GPU 0")
        manifest["checks"][condition + "_exact_adapter_tensors"] = True
        del saved, actual
        return current

    def generate(current, case):
        # The verified pilot path: nonthinking template, autocast FP16, one
        # logits position, synchronized timing, tokens and corrected EOS.
        current.eval()
        current.config.use_cache = True
        tokenized = tokenizer.apply_chat_template(case["messages"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False,
                    return_tensors="pt", return_dict=True).to("cuda:0")
        token_ids = tokenized["input_ids"][0].tolist()
        if token_ids != prompt_tokens[case["case_id"]]:
            raise RuntimeError("Actual generation input differs from frozen tokenization")
        set_seed(42)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            output = current.generate(**tokenized, generation_config=generation, logits_to_keep=1)
        torch.cuda.synchronize()
        ids = output[0, tokenized["input_ids"].shape[1]:].tolist()
        if not ids:
            raise RuntimeError("Generation returned no tokens")
        return {"token_ids": ids, "token_count": len(ids),
                "response": tokenizer.decode(ids, skip_special_tokens=True),
                "finish_reason": "stop" if ids[-1] == generation.eos_token_id else "length",
                "generation_seconds": time.perf_counter() - started,
                "input_sha256": fingerprint(token_ids), "input_token_ids": token_ids}

    def evaluate(current, condition):
        stage("generate_" + condition)
        rows = all_results[condition]
        for case in inputs[len(rows):]:
            row = {"identity_sha256": identity_sha, "condition": condition,
                   "case_id": case["case_id"], "category": case["category"], "split": case["split"],
                   "prompt_sha256": fingerprint(case["messages"]),
                   "generation_sha256": fingerprint(gen_dict), **generate(current, case)}
            atomic_journal(run_dir / f"{condition}_responses.jsonl", rows + [row])
            rows.append(row)
            print(condition, f"{len(rows)}/{len(inputs)}", case["case_id"], row["finish_reason"], flush=True)
        # Re-read the durable artifact using the same resume checks.
        all_results[condition] = read_evaluation_journal(run_dir / f"{condition}_responses.jsonl",
                identity_sha, condition, inputs, gen_dict, prompt_tokens)
        manifest["checks"][condition + "_complete"] = len(all_results[condition]) == len(inputs)
        memory(condition + "_complete")

    try:
        stage("verified_inputs_and_checkpoints")
        atomic_bytes(run_dir / "model_inputs.jsonl", input_bytes)
        atomic_bytes(run_dir / "runtime_protocol.json", protocol_bytes)
        atomic_json(run_dir / "comparison_manifest.json", {"identity_sha256": identity_sha,
            "conditions": list(CONDITIONS), "generation": gen_dict,
            "generation_sha256": fingerprint(gen_dict), "enable_thinking": False,
            "model_inputs_sha256": digest(input_bytes), "source_protocol_sha256": protocol["source_protocol_sha256"],
            "fresh_base_before_adapter_attachment": True,
            "step_500_reloads_base": True, "quality_scores_assigned": False})
        freeze = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], check=True,
                                 capture_output=True, text=True)
        atomic_bytes(run_dir / "pip_freeze.txt", freeze.stdout.encode())
        torch.cuda.reset_peak_memory_stats(0)
        stage("load_untouched_base")
        model = load_base()
        manifest["checks"]["fresh_base_has_no_adapter"] = True
        evaluate(model, "fresh_base")
        # Base answers are durably complete before any LoRA weights are attached.
        if not manifest["checks"]["fresh_base_complete"]:
            raise RuntimeError("Untouched baseline is incomplete")
        stage("load_step_50_adapter")
        model = attach(model, "step_50")
        evaluate(model, "step_50")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        memory("released_step_50")
        stage("fresh_reload_for_step_500")
        model = load_base()
        manifest["checks"]["step_500_fresh_base_reload"] = True
        model = attach(model, "step_500")
        evaluate(model, "step_500")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        summary = export_results(run_dir, all_results, inputs, identity_sha)
        if not summary["complete"] or not all(manifest["checks"].values()):
            raise RuntimeError("Comparison checks are incomplete")
        manifest["status"] = "complete"
        stage("complete")
        temporary = run_dir / ".comparison_delivery.zip.partial"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(run_dir.iterdir()):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".txt"}:
                    archive.write(path, path.name)
            for path in sorted(bundle_dir.iterdir()):
                if path.is_file():
                    archive.write(path, "inputs/" + path.name)
        os.replace(temporary, run_dir / "comparison_delivery.zip")
        return run_dir
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_json(run_dir / "run_manifest.json", manifest)
        export_results(run_dir, all_results, inputs, identity_sha)
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
