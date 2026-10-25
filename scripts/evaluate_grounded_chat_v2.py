"""Matched-source comparison of direct generation and evidence-first v2."""
import json
from pathlib import Path
try:
    from .grounded_chat import POLICY, fingerprint
    from .grounded_chat_v2 import EvidenceChat, initial_state, claims_for_trace, REVIEW_FAILURE
    from .kaggle_grounding import atomic_journal
except ImportError:
    from grounded_chat import POLICY, fingerprint
    from grounded_chat_v2 import EvidenceChat, initial_state, claims_for_trace, REVIEW_FAILURE
    from kaggle_grounding import atomic_journal

CONDITIONS = ('direct_shared_evidence', 'evidence_first_checked')


def model_spec(spec):
    return {'singles': [{'id': s['id'], 'question': s['question']} for s in spec['singles']],
            'conversations': [{'id': c['id'], 'turns': [{'question': t['question']} for t in c['turns']]} for c in spec['conversations']]}


def direct_request(collection, ledger, question, trace, history):
    cards = claims_for_trace(ledger, trace)
    # Both arms receive the same reviewed propositions and original packets.
    policy = POLICY.replace('e.g. [eastman]', 'using only the IDs supplied below')
    evidence = {'packets': [m['packet'] for m in trace['matches']], 'evidence_propositions': list(cards.values())}
    return [{'role': 'system', 'content': collection['persona_prompt']+'\n'+policy+'\n'+json.dumps(evidence, ensure_ascii=False)},
            *history[-6:], {'role': 'user', 'content': question}]


def read_journal(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Incomplete journal; preserve before inspecting recovery')
    return [json.loads(line) for line in raw.splitlines()]


def evaluate(collection, ledger, spec, generate, output, identity):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    bot = EvidenceChat(collection, ledger)
    calls_path, answers_path = output/'model_calls.jsonl', output/'answers.jsonl'
    old_calls, old_answers = read_journal(calls_path), read_journal(answers_path)
    calls, answers, paired_traces = [], [], {}
    for condition in CONDITIONS:
        jobs = [(s['id'], [s]) for s in spec['singles']]+[(c['id'], c['turns']) for c in spec['conversations']]
        for cid, turns in jobs:
            state, history = initial_state(collection, ledger), []
            for turn_index, turn in enumerate(turns, 1):
                key = f'{condition}/{cid}/{turn_index}'
                before = json.loads(json.dumps(state))
                trace = bot.retriever.retrieve(turn['question'], state)
                pair_key = (cid, turn_index)
                if condition == CONDITIONS[0]:
                    paired_traces[pair_key] = fingerprint(trace)
                elif paired_traces[pair_key] != fingerprint(trace):
                    raise ValueError('Matched arms received different retrieved evidence')
                step = 0
                def cached_generate(messages):
                    nonlocal step
                    step += 1
                    signature = fingerprint({'identity': identity, 'key': key, 'step': step, 'messages': messages})
                    if len(calls) < len(old_calls):
                        record = old_calls[len(calls)]
                        if (record.get('signature') != signature or record.get('messages') != messages
                                or record.get('output_sha256') != fingerprint(record.get('output'))):
                            raise ValueError('Model call journal differs from exact resume identity')
                    else:
                        result = generate(messages)
                        record = {'key': key, 'step': step, 'signature': signature, 'messages': messages,
                                  'output': result, 'output_sha256': fingerprint(result)}
                    calls.append(record)
                    if len(calls) > len(old_calls):
                        atomic_journal(calls_path, calls)
                    return record['output']
                if condition == CONDITIONS[0]:
                    result = cached_generate(direct_request(collection, ledger, turn['question'], trace, history))
                    normal = result.get('finish_reason') == 'stop' and isinstance(result.get('response'), str)
                    text = result['response'] if normal else REVIEW_FAILURE
                    state = {**state, 'active_packet_ids': trace['active_packet_ids'], 'turns': state['turns']+1}
                    answer = {'question': turn['question'], 'response': text, 'response_sha256': fingerprint(text),
                              'state_before': before, 'state_after': state, 'retrieval': trace,
                              'status': 'unfiltered_direct' if normal else 'blocked_incomplete',
                              'accepted_claim_ids': [], 'semantic_guarantee': False}
                else:
                    answer = bot.answer(turn['question'], cached_generate, state=state, history=history, trace=trace)
                    state = answer['state_after']
                record = {**answer, 'key': key, 'case_id': cid, 'turn': turn_index, 'condition': condition,
                          'model_calls': step, 'identity_sha256': fingerprint(identity)}
                if len(answers) < len(old_answers) and old_answers[len(answers)] != record:
                    raise ValueError('Answer journal differs on exact local replay')
                answers.append(record)
                if len(answers) > len(old_answers):
                    atomic_journal(answers_path, answers)
                history += [{'role': 'user', 'content': turn['question']}, {'role': 'assistant', 'content': record['response']}]
                print(key, record['status'], flush=True)
    if len(old_calls)>len(calls) or len(old_answers)>len(answers):
        raise ValueError('Trailing records in journal')
    return answers, calls
