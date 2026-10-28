"""Ten real conversational replies for human review; no model self-audit."""
import json
from pathlib import Path
from candidate_audit import run as run_candidate, validate_protocol as validate_common
from candidate_base_probe import CANDIDATE, sha
from grounded_chat import fingerprint
from grounded_request_boundary import ConversationState
from review_chat import prepare_turn, parse_reply, append_turn


def validate_protocol(p):
    if (p.get('purpose') != 'candidate_conversation_review' or p.get('conversation_count') != 5
            or p.get('maximum_displayed_answers') != 10 or p.get('model_audit_performed') is not False):
        raise ValueError('Human conversation review scope differs')
    expected_calls = 5 if p.get('seeded_reply_count') == 5 else 10
    if p.get('evaluation_calls') != expected_calls:
        raise ValueError('Unexpected new conversation call limit')
    if p.get('retrieval_policy', 'current') not in ('current', 'human_context_decay_v1'):
        raise ValueError('Unsupported conversation retrieval policy')
    validate_common({**p, 'purpose': 'candidate_neutral_audit', 'evaluation_calls': 10})


def evaluate(jobs, generate, output, profile, collection, ledger, tokenizer, today):
    if len(jobs) != 5 or len({j['case_id'] for j in jobs}) != 5 or sum(len(j['questions']) for j in jobs) != 10:
        raise ValueError('Expected five conversations and ten total turns')
    output = Path(output); output.mkdir(parents=True, exist_ok=True); rows = []; calls = []
    with (output/'answers.jsonl').open('x') as answers, (output/'model_calls.jsonl').open('x') as journal:
        for job in jobs:
            state = ConversationState(tuple(p['id'] for p in collection['packets'])); failed = False
            for turn, question in enumerate(job['questions']):
                request = prepare_turn(profile, collection, ledger, state, question, tokenizer=tokenizer,
                    reflection_allowed=job['reflection_allowed'], today=today,
                    retrieval_policy=job.get('retrieval_policy', 'current'))
                if turn == 0 and (request['messages'] != job['messages'] or request['input_token_ids'] != job['input_token_ids']):
                    raise ValueError('First conversation request differs from frozen inputs')
                error = None; reply = None
                seed = job.get('seed_reply') if turn == 0 else None
                if failed:
                    error = 'Prior reply could not be parsed; dependent turn was not generated.'
                elif seed is not None:
                    if seed['messages'] != request['messages'] or seed['output']['input_token_ids'] != request['input_token_ids']:
                        raise ValueError('Archived initial reply request differs')
                    reply = parse_reply(seed['output'], request)
                else:
                    out = generate(request['messages'])
                    call = {'case_id': job['case_id'], 'turn': turn+1, 'stage': 'conversation',
                            'messages': request['messages'], 'output': out, 'output_sha256': fingerprint(out)}
                    calls.append(call); journal.write(json.dumps(call, ensure_ascii=False)+'\n'); journal.flush()
                    try: reply = parse_reply(out, request)
                    except (ValueError, KeyError, TypeError) as exc: error = str(exc); failed = True
                row = {'case_id': job['case_id'], 'title': job['title'], 'turn': turn+1, 'question': question,
                       'status': 'ready_for_human_review' if reply else 'format_blocked',
                       'reply': reply, 'validation_error': error, 'request': request,
                       'model': CANDIDATE, 'quality_validated': False, 'human_review_status': 'unreviewed',
                       'model_audit_performed': False, 'active_model_changed': False,
                       'generation_origin': 'archived_saved47' if seed else 'new_model_call' if not error else 'blocked',
                       'training_steps': 0, 'adapters_loaded': [], 'memory_updates': []}
                rows.append(row); answers.write(json.dumps(row, ensure_ascii=False)+'\n'); answers.flush()
                if reply: state = append_turn(state, question, reply)
                print(job['case_id'], turn+1, row['status'], flush=True)
    return {'displayed_answers': len(rows), 'completed_evaluation_calls': len(calls),
            'answers_sha256': sha((output/'answers.jsonl').read_bytes()),
            'model_calls_sha256': sha((output/'model_calls.jsonl').read_bytes()), 'model_audit_performed': False}


class Experiment:
    def __init__(self, bundle): self.bundle = Path(bundle)
    def validate_protocol(self, p): validate_protocol(p)
    def evaluate(self, jobs, generate, output):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(CANDIDATE['model_id'], revision=CANDIDATE['revision'],
            local_files_only=True, trust_remote_code=False)
        profile, collection, ledger = [json.loads((self.bundle/n).read_text()) for n in ('profile.json','collection.json','ledger.json')]
        design = json.loads((self.bundle/'review_design.json').read_text())
        return evaluate(jobs, generate, output, profile, collection, ledger, tokenizer, design['current_date'])


def run(bundle, output): return run_candidate(bundle, output, experiment=Experiment(bundle))
