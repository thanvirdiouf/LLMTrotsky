#!/usr/bin/env python3
"""Audit proposed seed batches against cached sources without approving claims."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re

try:
    from .build_seed_dataset import (conversation_messages, date_evidence, normalized,
                                     read_rows, reserved_evaluation_families,
                                     source_blocks, validate_verbatim)
except ImportError:
    from build_seed_dataset import (conversation_messages, date_evidence, normalized,
                                    read_rows, reserved_evaluation_families,
                                    source_blocks, validate_verbatim)

CATEGORIES = {'adapted_passages', 'verbatim_passages', 'adapted_conversations'}


def audit(project, batch_paths):
    docs = {d['url']: d for d in read_rows(project / 'corpus_mia/documents.jsonl')}
    catalog = {d['document_id']: d for d in read_rows(project / 'corpus_mia/catalog.jsonl')}
    specs = json.loads((project / 'reviews/evaluation_specifications.json').read_text())
    reserved = reserved_evaluation_families(docs, catalog, specs)
    persona = json.loads((project / 'config/persona.json').read_text())
    existing = read_rows(project / 'datasets/seed_records.jsonl')
    questions = {normalized(r['messages'][1]['content']).casefold(): r['example_id'] for r in existing}
    identities = {(r['source_url'], r['messages'][1]['content']) for r in existing}
    answers = {normalized(r['messages'][-1]['content']).casefold(): r['example_id'] for r in existing if len(r['messages'][-1]['content'].split()) >= 12}
    checked_raw = set()
    records, errors, warnings = [], [], []
    for batch in batch_paths:
        data = json.loads(batch.read_text())
        for category in sorted(CATEGORIES):
            for index, item in enumerate(data.get(category, [])):
                ref = f'{batch.name}:{category}:{index}'
                try:
                    matches = [d for u, d in docs.items() if u.endswith(item['source_suffix'])]
                    if len(matches) != 1:
                        raise ValueError('Missing or ambiguous source suffix')
                    doc = matches[0]
                    if catalog[doc['document_id']]['source_family_id'] in reserved:
                        raise ValueError('Reserved evaluation family')
                    selected = source_blocks(doc, item['block_ids'])
                    temporal = date_evidence(doc, persona['historical_as_of'])
                    evidence = '\n\n'.join(b['text'] for b in selected)
                    if doc['document_id'] not in checked_raw:
                        raw = (project / 'corpus_mia' / doc['source_html']).read_bytes()
                        if hashlib.sha256(raw).hexdigest() != doc['source_sha256']:
                            raise ValueError('Raw source hash mismatch')
                        checked_raw.add(doc['document_id'])
                    messages = conversation_messages(persona, item['question'], item['answer'], item.get('turns'))
                    if category == 'adapted_conversations' and len(messages) < 5:
                        raise ValueError('Conversation proposal lacks a follow-up exchange')
                    if category == 'verbatim_passages':
                        for message in messages:
                            if message['role'] == 'assistant':
                                validate_verbatim(message['content'], evidence)
                    if not item.get('review_note') or not item.get('temporal_frame') or not item.get('genre'):
                        raise ValueError('Missing claim review, temporal frame or genre')
                    question = normalized(item['question']).casefold()
                    if question in questions or (doc['url'], item['question']) in identities:
                        raise ValueError('Duplicate question: ' + questions.get(question, ref))
                    for previous, previous_ref in questions.items():
                        ratio = SequenceMatcher(None, question, previous).ratio()
                        if ratio >= .86:
                            warnings.append({'item': ref, 'kind': 'similar_question_requires_review', 'other': previous_ref, 'ratio': round(ratio, 3)})
                    questions[question] = ref
                    identities.add((doc['url'], item['question']))
                    answer = normalized(item['answer']).casefold()
                    if len(item['answer'].split()) >= 12:
                        if answer in answers:
                            warnings.append({'item': ref, 'kind': 'same_final_answer_requires_review', 'other': answers[answer]})
                        answers[answer] = ref
                    record = {'item': ref, 'candidate_id': item.get('candidate_id'), 'source_url': doc['url'], 'source_sha256': doc['source_sha256'], 'source_block_ids': item['block_ids'], 'evidence_sha256': hashlib.sha256(evidence.encode()).hexdigest(), 'source_date_evidence': temporal, 'genre': item['genre'], 'multi_turn': len(messages) > 3, 'final_answer_words': len(item['answer'].split()), 'answer_origin': 'verbatim_source' if category == 'verbatim_passages' else 'adapted_source_grounded'}
                    records.append(record)
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append({'item': ref, 'error': str(exc)})
    return {'mechanical_status': 'failed' if errors else 'passed', 'proposed_records_checked': len(records), 'source_documents': len({r['source_url'] for r in records}), 'multi_turn': sum(r['multi_turn'] for r in records), 'genres': Counter(r['genre'] for r in records), 'errors': errors, 'warnings': warnings, 'records': records, 'claim_approval': False, 'limitations': 'Mechanical checks do not approve entailment, historical truth, attribution or conversation quality; a separate source review is required.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batches', nargs='+', type=Path)
    parser.add_argument('--project', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.project, args.batches)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, ensure_ascii=False, indent=2))
    if report['errors']:
        raise SystemExit(1)
