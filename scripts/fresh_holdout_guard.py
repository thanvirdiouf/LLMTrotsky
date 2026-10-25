#!/usr/bin/env python3
"""Additive source exclusions for frozen fresh holdouts.

This local authoring guard is separate from the historical pilot's immutable,
portable helper. It reads no model weights and rewrites no evaluation or seed
artifacts. Whole families are blocked even when only one family member appears
in an exclusion manifest. Pending holdouts without a protocol are not frozen.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def read_rows(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(f'Invalid fresh-holdout JSONL: {path}:{number}') from error
        if not isinstance(row, dict):
            raise ValueError(f'Expected source object: {path}:{number}')
        rows.append(row)
    return rows


@dataclass(frozen=True)
class FreshHoldoutGuard:
    document_ids: frozenset
    family_ids: frozenset
    manifests: tuple

    def check_source(self, document_id, family_id):
        if document_id in self.document_ids or family_id in self.family_ids:
            raise ValueError('Fresh evaluation holdout prohibits this training source: '
                             + str(document_id))


def load_fresh_holdout_guard(project, catalog):
    """Read all frozen additive reservations and reject stale/corrupt manifests."""
    project = Path(project)
    documents, families, manifests = set(), set(), []
    for directory in sorted((project / 'evaluation').glob('fresh_holdout_*')):
        if not directory.is_dir():
            continue
        path, protocol_path = directory / 'source_exclusions.jsonl', directory / 'protocol.json'
        if not path.exists() and not protocol_path.exists():
            continue  # Draft specifications alone do not reserve sources.
        if not path.is_file() or not protocol_path.is_file():
            raise ValueError('Incomplete frozen fresh holdout: ' + str(directory))
        try:
            protocol = json.loads(protocol_path.read_bytes())
            expected_hash = protocol['files']['source_exclusions.jsonl']
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError('Fresh holdout has no source-exclusion checksum: ' + str(directory)) from error
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError('Fresh source-exclusion checksum mismatch: ' + str(path))
        local_documents = set()
        for row in read_rows(path):
            document_id, family_id = row.get('document_id'), row.get('source_family_id')
            if (not isinstance(document_id, str) or not document_id
                    or not isinstance(family_id, str) or not family_id
                    or not isinstance(row.get('url'), str) or not row['url']
                    or row.get('training_prohibited') is not True
                    or document_id in local_documents):
                raise ValueError('Invalid/duplicate fresh source exclusion: ' + str(path))
            current = catalog.get(document_id)
            if (current is None or current.get('source_family_id') != family_id
                    or current.get('url') != row.get('url')):
                raise ValueError('Fresh source-exclusion catalog provenance is stale: ' + document_id)
            local_documents.add(document_id)
            documents.add(document_id)
            families.add(family_id)
        if not local_documents:
            raise ValueError('Frozen fresh source-exclusion manifest is empty: ' + str(path))
        manifests.append(path.relative_to(project).as_posix())
    return FreshHoldoutGuard(frozenset(documents), frozenset(families), tuple(manifests))


def check_training_records(records, catalog, guard):
    """Reject direct source leakage, family leakage, and forged/stale families."""
    for row in records:
        document_id = row.get('source_document_id')
        family_id = row.get('source_family_id')
        current = catalog.get(document_id)
        if (current is None or current.get('source_family_id') != family_id
                or (row.get('source_url') is not None
                    and current.get('url') != row['source_url'])):
            raise ValueError('Training source provenance differs from current catalog: ' + str(document_id))
        guard.check_source(document_id, family_id)


def guard_training_project(project, records_path=None):
    """Guard current source-bearing records before local notebook/lock staging."""
    project = Path(project)
    catalog_rows = read_rows(project / 'corpus_mia/catalog.jsonl')
    catalog = {row['document_id']: row for row in catalog_rows}
    if len(catalog) != len(catalog_rows):
        raise ValueError('Duplicate authoritative catalog document IDs')
    guard = load_fresh_holdout_guard(project, catalog)
    records = read_rows(records_path or project / 'datasets/seed_records.jsonl')
    check_training_records(records, catalog, guard)
    return {'records_checked': len(records), 'fresh_holdout_manifests': list(guard.manifests),
            'reserved_documents': len(guard.document_ids), 'reserved_families': len(guard.family_ids),
            'training_source_overlap': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--records', type=Path, help='Source-bearing reviewed candidate records to stage')
    args = parser.parse_args()
    print(json.dumps(guard_training_project(args.project, args.records), indent=2))
