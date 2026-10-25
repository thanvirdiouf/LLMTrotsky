#!/usr/bin/env python3
"""Guard fresh holdouts before deliberately staging a new training input lock.

The completed pilot's helper and lock remain unchanged. A new lock is written
only to an explicitly supplied, new path; the historical pilot lock cannot be
overwritten through this entrypoint. This command never launches training.
"""
import argparse
import json
from pathlib import Path

if __package__:
    from .build_training_pilot import build_lock, canonical_sha256
    from .fresh_holdout_guard import guard_training_project
else:
    from build_training_pilot import build_lock, canonical_sha256
    from fresh_holdout_guard import guard_training_project


def stage(project, output_lock=None, expected_records=500):
    project = Path(project)
    guard = guard_training_project(project)
    lock = build_lock(project, expected_records=expected_records)
    if output_lock is not None:
        target = Path(output_lock)
        if target.resolve() == (project / 'training/pilot_lock.json').resolve():
            raise ValueError('Preserve the historical pilot lock; select a new experiment path')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as stream:
            stream.write(json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    return {**guard, 'candidate_lock_sha256': canonical_sha256(lock),
            'lock_written': str(output_lock) if output_lock is not None else None,
            'training_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-lock', type=Path, help='New experiment lock path; existing files are refused')
    parser.add_argument('--expected-records', type=int, default=500)
    args = parser.parse_args()
    if args.expected_records <= 0:
        parser.error('--expected-records must be positive')
    print(json.dumps(stage(args.project, args.output_lock, args.expected_records), indent=2))
