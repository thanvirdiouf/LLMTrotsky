"""Inference-only 2x2 supplied-source grounding; Kaggle GPU0 only, no training."""
import hashlib
import json
import os
from pathlib import Path


CONDITIONS = ('base_without_passages','base_with_passages','adapter_without_passages','adapter_with_passages')
MODEL = {'model_id':'Qwen/Qwen3.5-9B','revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a'}
GENERATION = {'do_sample':False,'eos_token_id':248046,'max_new_tokens':512,'num_beams':1,'pad_token_id':248044,'use_cache':True}


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


def validate_payloads(input_bytes, protocol, control):
    if protocol['model'] != MODEL or protocol['generation'] != GENERATION or protocol['conditions'] != list(CONDITIONS):
        raise ValueError('Frozen inference policy changed')
    if protocol['cases'] != 24 or protocol['answers'] != 96 or protocol['training_steps'] != 0 or protocol['seed'] != 42 or protocol['enable_thinking'] is not False:
        raise ValueError('Invalid inference-only comparison')
    if digest(input_bytes) != protocol['files']['model_inputs.jsonl']:
        raise ValueError('Input checksum changed')
    if control['status'] != 'complete' or control['completed_steps'] != 50 or not control['checks'] or not all(v is True for v in control['checks'].values()):
        raise ValueError('Saved control incomplete')
    if control['identity'] != protocol['control_identity'] or fingerprint(control['identity']) != protocol['control_identity_sha256'] or control['identity_sha256'] != protocol['control_identity_sha256']:
        raise ValueError('Saved adapter provenance changed')
    identity=control['identity']
    if identity['settings']['learning_rate'] != 1e-4 or identity['packages'] != protocol['packages'] or identity['model_id'] != MODEL['model_id'] or identity['revision'] != MODEL['revision']:
        raise ValueError('Control model/stack changed')
    values=[json.loads(line) for line in input_bytes.splitlines()]
    if len(values) != 48:raise ValueError('Expected 48 case/treatment inputs')
    pairs={}
    for r in values:
        if set(r) != {'case_id','category','split','treatment','messages'} or r['split'] != 'fresh_grounding_v1' or r['treatment'] not in {'with_passages','without_passages'}:
            raise ValueError('Reference-bearing or invalid input row')
        if [m['role'] for m in r['messages']] != ['system','user'] or any(set(m) != {'role','content'} or not isinstance(m['content'],str) for m in r['messages']):
            raise ValueError('Invalid messages')
        if r['treatment'] in pairs.setdefault(r['case_id'],{}):raise ValueError('Duplicate input')
        pairs[r['case_id']][r['treatment']]=r
    if len(pairs)!=24 or any(set(v)!={'with_passages','without_passages'} for v in pairs.values()):raise ValueError('Incomplete matched inputs')
    for pair in pairs.values():
        no=pair['without_passages']['messages'];yes=pair['with_passages']['messages']
        if no[0]!=yes[0]:raise ValueError('Persona mismatch')
        import re
        removed=re.sub(r'<source_passages>\n.*?\n</source_passages>\n\n','',yes[1]['content'],flags=re.S)
        if removed != no[1]['content'] or '<source_passages>' in no[1]['content'] or yes[1]['content'].count('<source_passages>')!=1:
            raise ValueError('Only passage block may change')
    return values


def find_checkpoint(input_root, protocol, verify):
    paths=list(Path(input_root).rglob('checkpoint-000050'))
    if len(paths)!=1:raise ValueError('Expected one version-pinned saved control checkpoint')
    path=paths[0]
    manifest=verify(path,expected_identity=protocol['control_identity'])
    if manifest['completed_steps']!=50 or digest((path/'checkpoint_manifest.json').read_bytes())!=protocol['checkpoint_manifest_sha256']:
        raise ValueError('Saved checkpoint identity changed')
    for name,expected in [('adapter_model.safetensors',protocol['adapter_sha256']),('adapter_config.json',protocol['adapter_config_sha256'])]:
        if digest((path/'adapter'/name).read_bytes())!=expected:raise ValueError('Saved adapter bytes changed')
    return {'saved_lr1e4_step50':(path,manifest)}


def export_results(run_dir, all_results, inputs, identity_sha):
    expected=[r['case_id'] for r in inputs if r['treatment']=='without_passages']
    summary={'identity_sha256':identity_sha,'expected_per_condition':24,'completed_per_condition':{c:len(all_results[c]) for c in CONDITIONS},'complete':all([r['case_id'] for r in all_results[c]]==expected for c in CONDITIONS),'truncations_by_condition':{c:sum(r['finish_reason']=='length' for r in all_results[c]) for c in CONDITIONS},'quality_scores_assigned':False,'training_steps':0}
    atomic_json(Path(run_dir)/'evaluation_summary.json',summary)
    return summary


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
    pilot_bytes = (bundle_dir / "control_run_manifest.json").read_bytes()
    protocol, pilot_manifest = json.loads(protocol_bytes), json.loads(pilot_bytes)
    if digest(pilot_bytes) != protocol["control_run_manifest_sha256"]:
        raise ValueError("Archived pilot manifest checksum mismatch")
    inputs = validate_payloads(input_bytes, protocol, pilot_manifest)
    condition_inputs = {kind: [r for r in inputs if r["treatment"] == ("with_passages" if kind.endswith("_with_passages") else "without_passages")] for kind in CONDITIONS}
    from pilot_checkpoints import verify_checkpoint
    from kbit_preparation import prepare_frozen_quantized_base
    checkpoints = find_checkpoint(input_root, protocol, verify_checkpoint)
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
    prompt_tokens = {kind: tokenize_inputs(tokenizer, condition_inputs[kind], protocol["max_input_tokens"]) for kind in CONDITIONS}
    if prompt_tokens[CONDITIONS[0]] != prompt_tokens[CONDITIONS[2]] or prompt_tokens[CONDITIONS[1]] != prompt_tokens[CONDITIONS[3]]:
        raise ValueError("Matched treatments have different token IDs")
    if any(prompt_tokens[CONDITIONS[0]][c] == prompt_tokens[CONDITIONS[1]][c] for c in prompt_tokens[CONDITIONS[0]]):
        raise ValueError("Passage treatment is missing")
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
                "model_inputs_sha256": digest(input_bytes), "control_run_manifest_sha256": digest(pilot_bytes),
                "control_identity_sha256": protocol["control_identity_sha256"],
                "chat_template_sha256": template_hash, "generation": gen_dict,
                "generation_sha256": fingerprint(gen_dict), "prompt_token_ids_sha256": fingerprint(prompt_tokens),
                "helpers": {p.name: digest(p.read_bytes()) for p in sorted(bundle_dir.glob("*.py"))},
                "checkpoints": {kind: {"step": manifest["completed_steps"],
                    "manifest_sha256": digest((path / "checkpoint_manifest.json").read_bytes()),
                    "adapter_sha256": manifest["files"]["adapter/adapter_model.safetensors"],
                    "identity_sha256": manifest["identity_sha256"]}
                    for kind, (path, manifest) in checkpoints.items()},
                "purpose": "source_grounding_v1_inference_only", "optimizer_updates": 0}
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
                   kind, condition_inputs[kind], gen_dict, prompt_tokens[kind]) for kind in CONDITIONS}

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
        path, checkpoint = checkpoints["saved_lr1e4_step50"]
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

    def generate(current, case, condition):
        # The verified pilot path: nonthinking template, autocast FP16, one
        # logits position, synchronized timing, tokens and corrected EOS.
        current.eval()
        current.config.use_cache = True
        tokenized = tokenizer.apply_chat_template(case["messages"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False,
                    return_tensors="pt", return_dict=True).to("cuda:0")
        token_ids = tokenized["input_ids"][0].tolist()
        if token_ids != prompt_tokens[condition][case["case_id"]]:
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
        for case in condition_inputs[condition][len(rows):]:
            row = {"identity_sha256": identity_sha, "condition": condition,
                   "case_id": case["case_id"], "category": case["category"], "split": case["split"],
                   "prompt_sha256": fingerprint(case["messages"]),
                   "generation_sha256": fingerprint(gen_dict), **generate(current, case, condition)}
            atomic_journal(run_dir / f"{condition}_responses.jsonl", rows + [row])
            rows.append(row)
            print(condition, f"{len(rows)}/{len(condition_inputs[condition])}", case["case_id"], row["finish_reason"], flush=True)
        # Re-read the durable artifact using the same resume checks.
        all_results[condition] = read_evaluation_journal(run_dir / f"{condition}_responses.jsonl",
                identity_sha, condition, condition_inputs[condition], gen_dict, prompt_tokens[condition])
        manifest["checks"][condition + "_complete"] = len(all_results[condition]) == len(condition_inputs[condition])
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
            "matched_treatment_tokens_equal": True, "training_steps": 0, "quality_scores_assigned": False})
        freeze = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], check=True,
                                 capture_output=True, text=True)
        atomic_bytes(run_dir / "pip_freeze.txt", freeze.stdout.encode())
        torch.cuda.reset_peak_memory_stats(0)
        stage("load_untouched_base")
        model = load_base()
        manifest["checks"]["fresh_base_has_no_adapter"] = True
        evaluate(model, CONDITIONS[0])
        evaluate(model, CONDITIONS[1])
        if not all(manifest['checks'][c+'_complete'] for c in CONDITIONS[:2]):
            raise RuntimeError('Both untouched base arms must be complete before adapter loading')
        stage('load_saved_lr1e4_adapter')
        model = attach(model, 'saved_lr1e4_step50')
        evaluate(model, CONDITIONS[2])
        evaluate(model, CONDITIONS[3])
        del model
        gc.collect()
        torch.cuda.empty_cache()
        manifest['checks']['matched_treatment_tokens_equal'] = True
        manifest['checks']['zero_optimizer_updates'] = True
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
