"""CPU-only comparison of lexical source retention/release; zero model calls."""
import argparse
import json
from pathlib import Path
from grounded_chat import fingerprint
from grounded_request_boundary import ConversationState, PriorTurn
from review_chat import load_snapshot, prepare_turn

ROOT = Path(__file__).resolve().parents[1]


def audit(design, profile, collection, ledger):
    rows = []
    for case in design['cases']:
        turns = tuple(t for q in case['prior_questions'] for t in
                      (PriorTurn('user', q), PriorTurn('assistant', 'Unverified prior answer; not evidence.')))
        state = ConversationState(tuple(p['id'] for p in collection['packets']), prior_turns=turns)
        row = {'id': case['id'], 'kind': case['kind']}
        for policy in ('current', design['policy']):
            request = prepare_turn(profile, collection, ledger, state, case['question'], retrieval_policy=policy)
            ids = [p['id'] for p in request['source_packets']]
            checks = {'required_retrieved': case['required_packet'] in ids}
            if 'excluded_packet' in case:
                checks['previous_topic_absent'] = case['excluded_packet'] not in ids
            if 'excluded_top_packet' in case:
                checks['previous_topic_not_first'] = not ids or ids[0] != case['excluded_top_packet']
            row[policy] = {'packet_ids': ids, 'checks': checks, 'passed': all(checks.values()),
                           'trace': request['retrieval']}
        rows.append(row)
    return {'design_fingerprint': fingerprint(design), 'results': rows,
            'passed': {p: sum(row[p]['passed'] for row in rows) for p in ('current', design['policy'])},
            'quality_validated': False, 'model_calls': 0, 'active_route_changed': False,
            'limitation': 'Developer-authored source-membership controls only. No semantic answer evaluation; no real tokenizer budget test here.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', type=Path, default=ROOT/'config/ghost_conversation_retrieval_v1.json')
    p.add_argument('--output', type=Path, default=ROOT/'evaluation/runs/ghost_conversation_retrieval_v1/offline.json')
    a = p.parse_args(); design = json.loads(a.design.read_text())
    result = audit(design, *load_snapshot(ROOT/design['source_snapshot']))
    if a.output.exists():
        raise ValueError('Preserve the previous audit; choose a new output path')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': result['passed'], 'cases': len(result['results']), 'model_calls': 0}))
