"""Plan bounded evidence selections for explicit questions and follow-ups."""

import re

from conversation_retrieval import RESET
from grounded_request_boundary import ConversationState
from review_chat import prepare_turn
from reviewed_source_display import records_for_packets

INTERROGATIVE = (
    r"who|whose|what|which|why|how|when|where|did|does|do|was|were|is|are|"
    r"can|could|would|has|have|had"
)
BOUNDARY = re.compile(
    rf"(?:\?\s+|(?:,\s*)?\s+and\s+|;\s+)(?=(?:{INTERROGATIVE})\b)", re.I
)


def question_parts(question):
    """Split explicit interrogatives, with at most three calls per turn."""
    return [part.strip() for part in BOUNDARY.split(question, maxsplit=2) if part.strip()]


def plan_turn(profile, collection, ledger, question, prior_questions, tokenizer, today):
    state = ConversationState(tuple(packet['id'] for packet in collection['packets']))
    context = [] if RESET.search(question) else list(prior_questions[-4:])
    parts = []
    for part in question_parts(question):
        request = prepare_turn(
            profile, collection, ledger, state, part, tokenizer=tokenizer,
            reflection_allowed=False, today=today, retrieval_policy='human_context_decay_v1',
            select_records=True, prior_questions=context[-4:], source_limit=1,
        )
        packet_ids = [packet['id'] for packet in request['source_packets']]
        records = records_for_packets(collection, ledger, packet_ids) if packet_ids else {}
        request['allowed_record_ids'] = list(records)
        parts.append(request)
        context.append(part)
    return parts


def selection_requests(jobs):
    """Yield the actual model requests in their execution order."""
    for job in jobs:
        if 'selection_turns' in job:
            for parts in job['selection_turns']:
                for part in parts:
                    yield job, part
        else:
            for part in job.get('selection_parts', [job]):
                yield job, part


def evaluate(jobs, generate, output, profile, collection, ledger, tokenizer, today):
    """Replay retrieval and render selections for each turn from immutable records."""
    import json
    from pathlib import Path

    from candidate_base_probe import CANDIDATE, sha
    from grounded_chat import fingerprint
    from reviewed_source_display import render_selection, select_ids

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calls = displayed = 0
    with (output/'answers.jsonl').open('x') as answers, (output/'model_calls.jsonl').open('x') as journal:
        for job in jobs:
            for index, question in enumerate(job['questions']):
                parts = plan_turn(profile, collection, ledger, question, job['questions'][:index], tokenizer, today)
                if parts != job['selection_turns'][index]:
                    raise ValueError('Frozen follow-up plan differs')
                packets = {p['id']: p for part in parts for p in part['source_packets']}
                selected = {}
                passages = []
                part_results = []
                for part in parts:
                    packet_ids = [packet['id'] for packet in part['source_packets']]
                    records = records_for_packets(collection, ledger, packet_ids) if packet_ids else {}
                    if list(records) != part['allowed_record_ids']:
                        raise ValueError('Follow-up menu differs')
                    out = generate(part['messages'])
                    call = {'case_id': job['case_id'], 'turn': index+1, 'stage': 'conversation',
                            'messages': part['messages'], 'output': out, 'output_sha256': fingerprint(out)}
                    journal.write(json.dumps(call, ensure_ascii=False)+'\n')
                    journal.flush()
                    calls += 1
                    if out['finish_reason'] != 'stop':
                        raise ValueError('Incomplete follow-up selection')
                    ids = select_ids(out['response'], records)
                    part_results.append({'question': part['question'], 'packets': packet_ids, 'selected_claim_ids': ids})
                    if not ids:
                        passages.append('No reviewed statement selected for this part: ' + part['question'])
                        continue
                    new_ids = [record_id for record_id in ids if record_id not in selected]
                    if new_ids:
                        display = render_selection(new_ids, records)
                        passages.append(display['response'])
                        selected.update((record['id'], record) for record in display['source_records'])
                reply = {'reply': '\n\n'.join(passages), 'fictional_reflection': '',
                         'selected_claim_ids': list(selected), 'source_records': list(selected.values()),
                         'part_results': part_results, 'quality_validated': False}
                request = {**parts[0], 'question': question, 'source_packets': list(packets.values())}
                row = {'case_id': job['case_id'], 'title': job['title'], 'turn': index+1,
                       'question': question, 'reply': reply, 'request': request,
                       'status': 'ready_for_human_review', 'validation_error': None,
                       'model': CANDIDATE, 'human_review_status': 'unreviewed',
                       'quality_validated': False, 'model_audit_performed': False,
                       'active_model_changed': False, 'generation_origin': 'new_model_call',
                       'training_steps': 0, 'adapters_loaded': [], 'memory_updates': []}
                answers.write(json.dumps(row, ensure_ascii=False)+'\n')
                answers.flush()
                displayed += 1
                print(job['case_id'], index+1, 'ready_for_human_review', flush=True)
    return {'displayed_answers': displayed, 'completed_evaluation_calls': calls,
            'answers_sha256': sha((output/'answers.jsonl').read_bytes()),
            'model_calls_sha256': sha((output/'model_calls.jsonl').read_bytes()),
            'model_audit_performed': False}
