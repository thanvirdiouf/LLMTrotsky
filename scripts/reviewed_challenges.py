"""Select reviewed facts for a follow-up and keep any fictional take separate."""
import json
from pathlib import Path

from candidate_base_probe import CANDIDATE, sha
from constrained_selection import REPLY_STANCES
from grounded_chat import fingerprint, response_diagnostics
from grounded_chat_v2 import parse_object
from grounded_request_boundary import ConversationState
from review_chat import append_turn, parse_reply, prepare_turn
from reviewed_source_display import records_for_packets, render_selection, select_ids


def plan(job, profile, collection, ledger, tokenizer, today):
    state = ConversationState(tuple(p['id'] for p in collection['packets']))
    original, question = job['questions']
    settings = dict(tokenizer=tokenizer, today=today, retrieval_policy=job['retrieval_policy'])
    first = prepare_turn(profile, collection, ledger, state, original,
                         reflection_allowed=job['reflection_allowed'], include_summary_ids=False, **settings)
    seed = parse_reply({'response': json.dumps(job['scripted_reply']), 'finish_reason': 'stop'}, first)
    citations = seed['diagnostics']
    if len(citations['cited_packet_ids']) != 1 or citations['unknown_citation_ids']:
        raise ValueError('Scripted prior reply needs one citation supplied on its original turn')
    state = append_turn(state, original, seed)
    request = prepare_turn(profile, collection, ledger, state, question, reflection_allowed=False,
                           select_records=True, challenge_selection=True, source_limit=1,
                           prior_citations=citations['cited_packet_ids'], **settings)
    records = records_for_packets(collection, ledger, [p['id'] for p in request['source_packets']]) if request['source_packets'] else {}
    request['allowed_record_ids'] = list(records)
    return first, seed, request, records


def assemble(output, request, records):
    if output['finish_reason'] != 'stop':
        raise ValueError('Incomplete reviewed challenge selection')
    value = parse_object(output['response'])
    if set(value) != {'stance', 'claim_ids'} or value['stance'] not in REPLY_STANCES:
        raise ValueError('Invalid stance or reviewed challenge selection')
    ids = select_ids(json.dumps({'claim_ids': value['claim_ids']}), records)
    if len(ids) > 2 or (not ids and value['stance'] != 'answer'):
        raise ValueError('Invalid number of statements for this stance')
    display = render_selection(ids, records)
    lines = {
        'answer': [],
        'correct': ['I need to correct my earlier reply.'],
        'stand_by': ['I’d keep the distinction in my earlier reply.'],
    }[value['stance']]
    lines += [f"{r['statement']} [{r['source']['packet_id']}]" for r in display['source_records']]
    text = '\n\n'.join(lines) or "I couldn’t identify a reviewed statement that answers this in the supplied material."
    return {'reply': text, 'fictional_reflection': '', 'stance': value['stance'],
            'selected_claim_ids': ids, 'source_records': display['source_records'],
            'diagnostics': response_diagnostics(text, request['retrieval']),
            'generated_factual_text_used': False, 'quality_validated': False}


def reflection_messages(question, reply):
    statements = [r['statement'] for r in reply['source_records']]
    return [
        {'role': 'system', 'content': (
            'You are the fictional present-day ghost of Leon Trotsky. Give a brief imagined opinion '
            'addressing the user’s disagreement. State a principle or value judgment in your own voice. '
            'The factual answer is displayed separately. Do not narrate historical events, add factual '
            'claims, quotes or citations, or claim to speak for the historical person. Keep it under '
            '60 words. Return only JSON with one string field, fictional_reflection. '
            'Selected evidence for context: ' + json.dumps(statements, ensure_ascii=False)
        )},
        {'role': 'user', 'content': question},
    ]


def parse_reflection(output):
    raw = output['response']
    if output['finish_reason'] != 'stop' or '<think>' in raw or '</think>' in raw:
        raise ValueError('Incomplete or invalid reflection')
    value = parse_object(raw)
    text = value.get('fictional_reflection')
    if (set(value) != {'fictional_reflection'} or not isinstance(text, str)
            or not 1 <= len(text.split()) <= 80 or '[' in text or ']' in text):
        raise ValueError('Unexpected fictional reflection format')
    return text.strip()


def evaluate(jobs, generate, output, profile, collection, ledger, tokenizer, today):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calls = displayed = 0
    with (output/'answers.jsonl').open('x') as answers, (output/'model_calls.jsonl').open('x') as journal:
        def call(job, stage, messages):
            nonlocal calls
            out = generate(messages)
            entry = {'case_id': job['case_id'], 'turn': 2, 'stage': stage, 'messages': messages,
                     'output': out, 'output_sha256': fingerprint(out)}
            journal.write(json.dumps(entry, ensure_ascii=False)+'\n')
            journal.flush()
            calls += 1
            return out

        for job in jobs:
            first, seed, request, records = plan(job, profile, collection, ledger, tokenizer, today)
            if (first['messages'] != job['messages'] or first['input_token_ids'] != job['input_token_ids']
                    or request != job['challenge_request']):
                raise ValueError('Frozen reviewed challenge changed')
            reply = assemble(call(job, 'conversation', request['messages']), request, records)
            if job['reflection_allowed']:
                out = call(job, 'reflection', reflection_messages(request['question'], reply))
                try:
                    reply['fictional_reflection'] = parse_reflection(out)
                    reply['reflection_status'] = 'ready_for_human_review'
                except ValueError:
                    reply['reflection_status'] = 'format_blocked'
            for turn, answer, ask, origin in (
                (1, seed, first, 'scripted_challenge'),
                (2, reply, request, 'reviewed_evidence_selection'),
            ):
                row = {'case_id': job['case_id'], 'title': job['title'], 'turn': turn, 'question': ask['question'],
                       'status': 'ready_for_human_review', 'reply': answer, 'validation_error': None, 'request': ask,
                       'model': CANDIDATE, 'quality_validated': False, 'human_review_status': 'unreviewed',
                       'model_audit_performed': False, 'active_model_changed': False, 'generation_origin': origin,
                       'training_steps': 0, 'adapters_loaded': [], 'memory_updates': []}
                answers.write(json.dumps(row, ensure_ascii=False)+'\n')
                answers.flush()
                displayed += 1
            print(job['case_id'], 'reviewed_evidence_selection', flush=True)
    return {'displayed_answers': displayed, 'completed_evaluation_calls': calls,
            'answers_sha256': sha((output/'answers.jsonl').read_bytes()),
            'model_calls_sha256': sha((output/'model_calls.jsonl').read_bytes()), 'model_audit_performed': False}
