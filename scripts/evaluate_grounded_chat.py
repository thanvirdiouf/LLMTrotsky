#!/usr/bin/env python3
"""Run fresh-turn retrieval and conversation evaluation with a supplied backend."""
import json
from pathlib import Path
try:
    from .grounded_chat import make_request, fingerprint, response_diagnostics, Retriever
except ImportError:
    from grounded_chat import make_request, fingerprint, response_diagnostics, Retriever


def model_spec(spec):
    """Keep reference judgments/checklists outside inference payloads."""
    return {'singles': [{k: s[k] for k in ('id', 'question', 'oracle_packet_ids')} for s in spec['singles']],
            'conversations': [{'id': c['id'], 'turns': [{'question': t['question']} for t in c['turns']]}
                              for c in spec['conversations']]}


def evaluate(collection, spec, generate, output, identity):
    """Replay an exact journal prefix and feed actual generated replies forward."""
    try:
        from .kaggle_grounding import atomic_journal
    except ImportError:
        from kaggle_grounding import atomic_journal
    retriever = Retriever(collection)
    output = Path(output)
    old = []
    if output.exists():
        payload = output.read_bytes()
        if payload and not payload.endswith(b'\n'):
            raise ValueError('Incomplete journal; preserve and inspect before recovery')
        old = [json.loads(line) for line in payload.splitlines()]
    rows = []
    jobs = [('retrieved', s['id'], [s]) for s in spec['singles']]
    jobs += [('oracle', s['id'], [s]) for s in spec['singles']]
    jobs += [('conversation', c['id'], c['turns']) for c in spec['conversations']]
    for condition, case_id, turns in jobs:
        history = []
        for index, turn in enumerate(turns):
            key = f'{condition}/{case_id}/{index+1}'
            request = make_request(collection, retriever, turn['question'], history,
                oracle_ids=turn['oracle_packet_ids'] if condition == 'oracle' else None)
            signature = fingerprint({'identity': identity, 'key': key, 'prompt_sha256': request['prompt_sha256'],
                                     'retrieval': request['retrieval']})
            if len(rows) < len(old):
                row = old[len(rows)]
                if (row.get('signature') != signature or row.get('key') != key
                        or row.get('request') != request or not isinstance(row.get('response'), str)
                        or row.get('response_sha256') != fingerprint(row['response'])
                        or row.get('finish_reason') not in {'stop', 'length'}):
                    raise ValueError('Resume journal differs from exact request/response identity')
            else:
                answer = generate(request['messages'])
                if not isinstance(answer.get('response'), str) or answer.get('finish_reason') not in {'stop', 'length'}:
                    raise ValueError('Backend must return text and stop/length diagnostics')
                row = {**answer, 'key': key, 'case_id': case_id, 'turn': index+1, 'condition': condition,
                       'signature': signature, 'identity_sha256': fingerprint(identity), 'request': request,
                       'response_sha256': fingerprint(answer['response']),
                       'diagnostics': response_diagnostics(answer['response'], request['retrieval'])}
            rows.append(row)
            if len(rows) > len(old):
                atomic_journal(output, rows)
            if row['finish_reason'] != 'stop':
                # A truncated reply stays visible in the journal but cannot silently
                # become the premise for a sustained conversation.
                raise ValueError('Truncated generation; evaluation incomplete, inspect before continuing')
            history.extend([{'role': 'user', 'content': turn['question']},
                            {'role': 'assistant', 'content': row['response']}])
            print(key, 'complete', flush=True)
    if len(old) > len(rows):
        raise ValueError('Unexpected trailing journal rows')
    return rows


def retrieval_preflight(collection, spec):
    r = Retriever(collection)
    return [{'case_id': s['id'], 'expected_packet_ids': s['oracle_packet_ids'],
             'selected_packet_ids': [m['packet']['id'] for m in r.retrieve(s['question'])['matches']],
             'necessary_packet_recalled': set(s['oracle_packet_ids']) <= {m['packet']['id'] for m in r.retrieve(s['question'])['matches']},
             'answerability': 'requires_source_review'} for s in spec['singles']]
