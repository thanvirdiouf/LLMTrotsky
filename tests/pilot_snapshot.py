"""Materialize only the input inventory of the historical frozen pilot for tests.

The live project deliberately gains fresh evaluation artifacts. Its historical
lock must reject that new inventory, while its archived notebook must remain
reproducible from the exact originally frozen files. This fixture never edits a
lock or silently removes live artifacts.
"""
import hashlib
import json
from pathlib import Path
import shutil


def materialize_pilot_snapshot(project, destination, bundled_paths):
    project, destination = Path(project), Path(destination)
    lock = json.loads((project / 'training/pilot_lock.json').read_bytes())
    for relative, expected in lock['input_sha256'].items():
        source = project / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Historical frozen input changed: ' + relative)
    paths = set(lock['input_sha256']) | set(bundled_paths) | {
        'training/pilot_lock.json', 'training/pilot_protocol.json',
        'training/pilot_evaluation_inputs.jsonl',
        'training/runs/kaggle_361399211/run_manifest.json',
        'training/runs/kaggle_361399211/tokenization_report.json',
    }
    for relative in sorted(paths):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, target)
    return destination
