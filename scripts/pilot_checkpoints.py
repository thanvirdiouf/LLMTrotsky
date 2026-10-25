"""Atomic, verified checkpoints for the one-epoch QLoRA pilot.

Importing this module needs only the standard library. Training state uses
``torch.load(..., weights_only=True)``; Python and optional NumPy RNG state use
JSON. A completed checkpoint is published by one directory rename, after its
adapter, optimizer, scaler, RNG state and optional metrics have been hashed.

``step`` is both completed optimizer updates and the next index in the frozen
one-example-per-update order. Call only at an update boundary, after writing the
corresponding metrics row. Restore into an already prepared, compatible model
and optimizer; this helper never constructs a base model or another adapter.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import uuid


SCHEMA_VERSION = 1
MANIFEST = "checkpoint_manifest.json"
COMPLETE = "COMPLETE"
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d{6})")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_inputs(identity, order_ids):
    if not isinstance(identity, dict) or not identity:
        raise ValueError("Checkpoint identity must be a nonempty JSON object")
    # JSON roundtrip prevents mutable caller references or exotic key types.
    identity = json.loads(_json_bytes(identity))
    order_ids = list(order_ids)
    if (not order_ids or any(not isinstance(key, str) or not key for key in order_ids)
            or len(set(order_ids)) != len(order_ids)):
        raise ValueError("Checkpoint order must contain unique nonempty example IDs")
    return identity, order_ids


def _write_file(path, payload):
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inventory(path):
    files = {}
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError("Checkpoint may not contain symbolic links")
        if child.is_file() and child.name not in {MANIFEST, COMPLETE}:
            files[child.relative_to(path).as_posix()] = _file_sha256(child)
    return files


def _tuple_tree(value):
    return tuple(_tuple_tree(child) for child in value) if isinstance(value, list) else value


def _numpy_state(numpy):
    if numpy is None:
        return None
    algorithm, keys, position, has_gauss, cached_gaussian = numpy.random.get_state()
    return {"algorithm": algorithm, "keys": keys.tolist(), "position": int(position),
            "has_gauss": int(has_gauss), "cached_gaussian": float(cached_gaussian)}


def _verify_metrics(payload, step):
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    except (UnicodeError, ValueError) as error:
        raise ValueError("Checkpoint metrics must be valid JSONL") from error
    if len(rows) != step or any(not isinstance(row, dict)
            or row.get("verified_optimizer_step") != index
            for index, row in enumerate(rows, 1)):
        raise ValueError("Checkpoint metrics must contain exactly the completed optimizer steps")


def verify_checkpoint(checkpoint_dir, expected_identity=None, order_ids=None):
    """Return a verified manifest; reject corrupt or incompatible checkpoints.

    Validation is read-only and happens before any model or RNG mutation. Hashes
    detect accidental corruption and mixed run files, rather than authenticate
    externally supplied checkpoints; resume only artifacts from your own run.
    """
    path = Path(checkpoint_dir)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Checkpoint must be a real directory")
    try:
        manifest_bytes = (path / MANIFEST).read_bytes()
        marker = json.loads((path / COMPLETE).read_bytes())
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as error:
        raise ValueError("Checkpoint is incomplete or has invalid metadata") from error
    if (not isinstance(marker, dict) or marker.get("manifest_sha256") != _sha256(manifest_bytes)
            or not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION):
        raise ValueError("Checkpoint completion marker/schema mismatch")
    try:
        identity, saved_order = _frozen_inputs(manifest["identity"], manifest["order_ids"])
        step = manifest["completed_steps"]
        if (isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= len(saved_order)
                or manifest["next_index"] != step
                or manifest["records"] != len(saved_order)
                or manifest["identity_sha256"] != _sha256(_json_bytes(identity))
                or manifest["order_sha256"] != _sha256(_json_bytes(saved_order))):
            raise ValueError("Checkpoint progress/identity/order mismatch")
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match and int(match[1]) != step:
            raise ValueError("Checkpoint directory name differs from its progress")
        files = manifest["files"]
        if (not isinstance(files, dict) or manifest["inventory_sha256"] != _sha256(_json_bytes(files))
                or files != _inventory(path)):
            raise ValueError("Checkpoint file inventory/hash mismatch")
        required = {"adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                    "training_state.pt", "rng_state.json"}
        if not required <= set(files):
            raise ValueError("Checkpoint is missing adapter or training state")
        if "training_metrics.jsonl" in files:
            _verify_metrics((path / "training_metrics.jsonl").read_bytes(), step)
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint metadata is malformed") from error
    if expected_identity is not None:
        if json.loads(_json_bytes(expected_identity)) != identity:
            raise ValueError("Checkpoint run identity/settings differ from this run")
    if order_ids is not None and list(order_ids) != saved_order:
        raise ValueError("Checkpoint example order differs from this run")
    return manifest


def find_latest_checkpoint(run_dir, expected_identity=None, order_ids=None):
    """Find the highest published step, failing if that checkpoint is corrupt.

    Unpublished ``.partial`` directories are ignored. There is deliberately no
    fallback from a corrupt latest checkpoint to an older update boundary.
    """
    directory = Path(run_dir)
    if not directory.exists():
        return None
    candidates = [path for path in directory.iterdir()
                  if CHECKPOINT_PATTERN.fullmatch(path.name)]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: int(CHECKPOINT_PATTERN.fullmatch(path.name)[1]))
    verify_checkpoint(latest, expected_identity, order_ids)
    return latest


def save_checkpoint(run_dir, step, identity, order_ids, model, optimizer, scaler,
                    torch=None, *, save_state=None, numpy=None, metrics_path=None):
    """Publish an immutable checkpoint and return its directory.

    ``save_state`` can replace ``torch.save`` for tests; it receives ``(state,
    path)``. Explicit ``numpy`` overrides the already-imported NumPy module, if
    any. Saving never imports NumPy and never saves embedding/base weights.
    """
    identity, order_ids = _frozen_inputs(identity, order_ids)
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= len(order_ids):
        raise ValueError("Checkpoint step must be a completed index in this one-epoch order")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    latest = find_latest_checkpoint(directory, identity, order_ids)
    if latest is not None and int(CHECKPOINT_PATTERN.fullmatch(latest.name)[1]) >= step:
        raise FileExistsError("Refusing to overwrite or rewind a completed checkpoint")
    torch = torch if torch is not None else importlib.import_module("torch")
    numpy = numpy if numpy is not None else sys.modules.get("numpy")
    rng = {"python": random.getstate(), "numpy": _numpy_state(numpy)}
    state = {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "torch_cpu_rng": torch.get_rng_state(),
             "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    metrics = Path(metrics_path).read_bytes() if metrics_path is not None else None
    if metrics is not None:
        _verify_metrics(metrics, step)
    destination = directory / f"checkpoint-{step:06d}"
    temporary = directory / f".{destination.name}.{uuid.uuid4().hex}.partial"
    temporary.mkdir()
    try:
        adapter_dir = temporary / "adapter"
        adapter_dir.mkdir()
        model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
        (save_state or torch.save)(state, temporary / "training_state.pt")
        _write_file(temporary / "rng_state.json", _json_bytes(rng))
        if metrics is not None:
            _write_file(temporary / "training_metrics.jsonl", metrics)
        files = _inventory(temporary)
        manifest = {"schema_version": SCHEMA_VERSION, "completed_steps": step,
                    "next_index": step, "records": len(order_ids),
                    "identity": identity, "identity_sha256": _sha256(_json_bytes(identity)),
                    "order_ids": order_ids, "order_sha256": _sha256(_json_bytes(order_ids)),
                    "files": files, "inventory_sha256": _sha256(_json_bytes(files))}
        manifest_bytes = _json_bytes(manifest)
        _write_file(temporary / MANIFEST, manifest_bytes)
        _write_file(temporary / COMPLETE, _json_bytes({"manifest_sha256": _sha256(manifest_bytes)}))
        verify_checkpoint(temporary, identity, order_ids)
        # Flush library-created adapter/state files too before the atomic rename.
        for filename in files:
            with (temporary / filename).open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(adapter_dir)
        _sync_directory(temporary)
        if destination.exists():
            raise FileExistsError("Checkpoint destination appeared during saving")
        temporary.rename(destination)
        _sync_directory(directory)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _load_existing_adapter(model, adapter_dir):
    from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    expected = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    if set(state) != set(expected) or any(state[key].shape != expected[key].shape for key in state):
        raise ValueError("Checkpoint adapter keys/shapes differ from the prepared model")
    result = set_peft_model_state_dict(model, state, adapter_name="default",
                                       ignore_mismatched_sizes=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys during restore: {result.unexpected_keys}")
    restored = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    if any(not restored[key].detach().cpu().equal(state[key]) for key in state):
        raise ValueError("Adapter tensors did not restore exactly")


def load_checkpoint(checkpoint_dir, expected_identity, order_ids, model, optimizer,
                    scaler, torch=None, *, load_adapter=None, load_state=None, numpy=None):
    """Restore a verified update boundary into existing objects; return its step.

    ``load_adapter(model, adapter_dir)`` and ``load_state(path, map_location=...,
    weights_only=...)`` are injectable for CPU tests. Training state is always
    loaded on CPU with restricted deserialization. RNG restoration happens last,
    after adapter/optimizer/scaler loading has completed.
    """
    manifest = verify_checkpoint(checkpoint_dir, expected_identity, order_ids)
    path = Path(checkpoint_dir)
    torch = torch if torch is not None else importlib.import_module("torch")
    state = (load_state or torch.load)(path / "training_state.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"optimizer", "scaler", "torch_cpu_rng", "torch_cuda_rng"}:
        raise ValueError("Checkpoint training state has unsupported fields")
    rng = json.loads((path / "rng_state.json").read_bytes())
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy"}:
        raise ValueError("Checkpoint RNG state has unsupported fields")
    saved_cuda = state["torch_cuda_rng"]
    current_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(saved_cuda) != current_devices:
        raise ValueError("Checkpoint CUDA RNG device count differs from this run")
    # Validate Python state on an independent generator before model mutation.
    random.Random().setstate(_tuple_tree(rng["python"]))
    if rng["numpy"] is not None:
        numpy = numpy if numpy is not None else sys.modules.get("numpy")
        if numpy is None:
            numpy = importlib.import_module("numpy")
        saved_numpy = rng["numpy"]
        numpy_state = (saved_numpy["algorithm"], numpy.array(saved_numpy["keys"], dtype="uint32"),
                       saved_numpy["position"], saved_numpy["has_gauss"], saved_numpy["cached_gaussian"])
        numpy.random.RandomState().set_state(numpy_state)
    elif numpy is not None or "numpy" in sys.modules:
        raise ValueError("Checkpoint has no NumPy RNG state but this run uses NumPy")
    (load_adapter or _load_existing_adapter)(model, path / "adapter")
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if saved_cuda:
        torch.cuda.set_rng_state_all(saved_cuda)
    if rng["numpy"] is not None:
        numpy.random.set_state(numpy_state)
    random.setstate(_tuple_tree(rng["python"]))
    return manifest["completed_steps"]
