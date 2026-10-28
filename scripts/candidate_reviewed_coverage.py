"""Five ID-only selections organised by literal question parts; no truth gate."""
import json
from pathlib import Path
from candidate_audit import run as run_candidate, validate_protocol as validate_common
from candidate_base_probe import CANDIDATE, sha
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError, parse_object
from reviewed_source_display import records_for_packets, select_ids, render_selection, MAX_SELECTED


def validate_protocol(p):
    exact = {'purpose': 'candidate_reviewed_coverage', 'evaluation_calls': 5,
             'independent_questions': 5, 'maximum_displayed_answers': 5,
             'maximum_selected_records': MAX_SELECTED, 'generated_factual_text_used': False,
             'model_audit_performed': False, 'part_presence_is_semantic_coverage': False}
    if any(p.get(k) != v or type(p.get(k)) is not type(v) for k, v in exact.items()):
        raise ValueError('Reviewed coverage scope differs')
    validate_common({**p, 'purpose': 'candidate_neutral_audit', 'evaluation_calls': 10})


def select_parts(text, records, count):
    parsed = parse_object(text)
    if set(parsed) != {'part_claim_ids'} or not isinstance(parsed['part_claim_ids'], list):
        raise EvidenceError('Selection may contain question-part ID arrays only')
    groups = parsed['part_claim_ids']
    if len(groups) != count:
        raise EvidenceError('One ID array is required per literal question part')
    ids = []
    for group in groups:
        for cid in select_ids(json.dumps({'claim_ids': group}), records):
            if cid not in ids:
                ids.append(cid)
    # Repeated IDs across parts are legitimate; the displayed union is bounded.
    select_ids(json.dumps({'claim_ids': ids}), records)
    return groups, ids


def evaluate(jobs, generate, output, collection, ledger):
    if len(jobs) != 5 or len({j['case_id'] for j in jobs}) != 5:
        raise ValueError('Expected five original independent questions')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows, calls = [], []
    with (output/'answers.jsonl').open('x') as answers, (output/'model_calls.jsonl').open('x') as journal:
        for job in jobs:
            records = records_for_packets(collection, ledger, job['source_packet_ids'])
            if {cid: r['record_fingerprint'] for cid, r in records.items()} != job['available_record_fingerprints']:
                raise ValueError('Reviewed menu differs from frozen inputs')
            if not job['request_parts'] or ''.join(job['request_parts']) != job['question']:
                raise ValueError('Literal parts do not preserve the original question')
            out = generate(job['messages'])
            call = {'case_id': job['case_id'], 'stage': 'select_parts', 'messages': job['messages'],
                    'output': out, 'output_sha256': fingerprint(out)}
            calls.append(call); journal.write(json.dumps(call, ensure_ascii=False)+'\n'); journal.flush()
            error = None; groups = []
            try:
                if out.get('finish_reason') != 'stop':
                    raise EvidenceError('Incomplete or truncated part selection')
                groups, ids = select_parts(out['response'], records, len(job['request_parts']))
                display = render_selection(ids, records)
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc); display = render_selection([], records)
                display['response'] = 'I could not select reviewed source summaries in the required format. The passages may contain relevant evidence.'
                display['response_fingerprint'] = fingerprint(display['response'])
            row = {**display, 'case_id': job['case_id'], 'question': job['question'], 'model': CANDIDATE,
                   'application_mode': 'experimental_reviewed_question_part_selection',
                   'source_context_packet_ids': job['source_packet_ids'], 'request_parts': job['request_parts'],
                   'part_claim_ids': groups, 'part_presence_is_semantic_coverage': False,
                   'status': 'blocked' if error else 'reviewed_summaries_selected' if display['selected_claim_ids'] else 'no_summaries_selected',
                   'validation_error': error, 'reference_date': collection['reference_date'],
                   'training_steps': 0, 'adapters_loaded': []}
            rows.append(row); answers.write(json.dumps(row, ensure_ascii=False)+'\n'); answers.flush()
            print(job['case_id'], row['status'], len(row['selected_claim_ids']), flush=True)
    return {'displayed_answers': len(rows), 'completed_evaluation_calls': len(calls),
            'answers_sha256': sha((output/'answers.jsonl').read_bytes()),
            'model_calls_sha256': sha((output/'model_calls.jsonl').read_bytes()),
            'generated_factual_text_used': False, 'model_audit_performed': False}


class Experiment:
    def __init__(self, bundle): self.bundle = Path(bundle)
    def validate_protocol(self, p): validate_protocol(p)
    def evaluate(self, jobs, generate, output):
        return evaluate(jobs, generate, output, *[json.loads((self.bundle/n).read_text()) for n in ('collection.json', 'ledger.json')])


def run(bundle, output):
    return run_candidate(bundle, output, experiment=Experiment(bundle))
