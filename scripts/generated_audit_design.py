"""Post-hoc checker diagnostic; generated candidates remain ignored locally."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from build_ghost_candidate_probe import enc,sha
from build_ghost_smoke import model_cases

DESIGN='config/ghost_generated_audit_design_v1.json'


def pair_groups(scores):
    pairs=scores['pairs']
    if len(pairs)!=5 or [p['pair_id'] for p in pairs[:2]]!=['d01_resident_field','d02_self_correction'] or [p['pair_id'] for p in pairs[2:]]!=['p01','p03','p05']:
        raise ValueError('Diagnostic score groups differ')
    return {name:{'pairs_both_correct':sum(p['both_decisions_match'] for p in items),'pairs_total':len(items)}
        for name,items in [('new_hearing_failures',pairs[:2]),('familiar_controls_with_added_context',pairs[2:])]}


def assemble_spec(design,answers,baseline):
    if design['schema_version']!=1 or design['source_packet_ids']!=['hearing','rakovsky','bessedovsky']:
        raise ValueError('Unknown generated-audit design')
    origin=design['generated_pair'];answer=next(a for a in answers if a['case_id']==origin['case_id'])
    text=next(s['text'] for s in answer['segments'] if s['id']==origin['segment_id'])
    if sha(text.encode())!=origin['text_sha256']:raise ValueError('Generated candidate changed')
    cases=[];pairs=[]
    def pair(identifier,question,bad,good,rationale):
        ids=[]
        for suffix,wording,approved in [('01',bad,False),('02',good,True)]:
            cid=identifier+'_'+suffix;ids.append(cid)
            cases.append({'id':cid,'question':question,'claim_ids':[],
                'segments':[{'id':'s1','kind':'source_analysis','text':wording}],
                'expected_approved':approved,'local_rationale':rationale})
        pairs.append({'pair_id':identifier,'case_ids':ids})
    pair(origin['pair_id'],origin['question'],text,origin['corrected_text'],origin['rationale'])
    authored=design['authored_pair'];pair(authored['pair_id'],authored['question'],authored['incorrect_text'],authored['corrected_text'],authored['rationale'])
    if design['control_pair_ids']!=['p01','p03','p05']:raise ValueError('Control design changed')
    byid={c['id']:c for c in baseline['cases']}
    for pid in design['control_pair_ids']:
        old=next(p for p in baseline['reviewer_only_pairs'] if p['pair_id']==pid)
        cases.extend(deepcopy(byid[i]) for i in old['case_ids']);pairs.append(deepcopy(old))
    if len(cases)!=10 or len({c['id'] for c in cases})!=10:raise ValueError('Diagnostic case coverage differs')
    return {'schema_version':1,'reference_date':design['reference_date'],'status':design['status'],
        'audit_calibration':{'policy':'neutral_source','packet_ids':design['source_packet_ids']},'conversation':[],
        'cases':cases,'reviewer_only_pairs':pairs,'control_context_note':design['control_context_note']}


def requests(values,spec):
    """Render exact old neutral prompt with frozen source packets, no retriever."""
    code='''import json
from pathlib import Path
from source_audit import source_audit_request
p=Path('.');spec=json.loads((p/'diagnostic_spec.json').read_text());collection=json.loads((p/'collection.json').read_text());profile=json.loads((p/'profile.json').read_text())
byid={p['id']:p for p in collection['packets']};trace={'matches':[{'packet':byid[i]} for i in spec['audit_calibration']['packet_ids']]}
rows=[{'case_id':c['id'],'messages':source_audit_request(profile,c['question'],trace,c['segments'])['messages']} for c in spec['cases']]
(p/'requests.json').write_text(json.dumps(rows))
'''
    with tempfile.TemporaryDirectory(prefix='generated_audit_requests_') as temp:
        d=Path(temp)
        for n,raw in values.items():(d/n).write_bytes(raw)
        # Only model-visible fields are needed to construct requests.
        (d/'diagnostic_spec.json').write_bytes(enc(model_cases(spec)))
        subprocess.run([sys.executable,'-c',code],cwd=d,check=True,capture_output=True,text=True,timeout=60)
        return json.loads((d/'requests.json').read_text())


def load(root,values,baseline,tokenizer=None):
    raw=(root/DESIGN).read_bytes();design=json.loads(raw);source=design['source_run'];version=source['local_version']
    if version!='ghost_candidate_synthesis_v2' or source['saved_version']!=41:raise ValueError('Wrong diagnostic origin')
    from verify_ghost_candidate_synthesis import verify
    verified=verify(version,root,tokenizer)
    if verified['status']!='verified' or any(verified[k]!=source[k] for k in ('result_sha256','answers_sha256')):raise ValueError('Diagnostic origin not verified')
    run=root/'evaluation/runs'/version;review=json.loads((run/'SOURCE_REVIEW.json').read_text())
    if any(review[k]!=verified[k] for k in ('result_sha256','answers_sha256')) or review['fresh_transfer_justified'] is not False:raise ValueError('Diagnostic source review absent')
    answers=[json.loads(l) for l in (run/'download/ghost_candidate_synthesis_results/answers.jsonl').read_text().splitlines()]
    spec=assemble_spec(design,answers,baseline)
    return spec,enc(spec),requests(values,spec),{'blueprint_path':DESIGN,'blueprint_sha256':sha(raw),'source_run':source,
        'context_policy':'Same three full packets for all ten cases; old controls have added hearing context',
        'parser_policy':'archived_neutral_audit_coverage_v1'}


def replay(values,spec,rows):
    """Use archived parsing/coverage semantics without a two-packet retriever."""
    code='''import json
from pathlib import Path
from ghost_chat import complete_review_coverage
from grounded_chat_v2 import parse_object,validate_review
p=Path('.');cases=json.loads((p/'diagnostic_spec.json').read_text())['cases'];rows=json.loads((p/'recorded_rows.json').read_text());answers=[]
if len(cases)!=len(rows):raise ValueError('Diagnostic parser coverage differs')
for case,row in zip(cases,rows):
 if row['case_id']!=case['id']:raise ValueError('Diagnostic parser order differs')
 out=row['output'];accepted=set();error=None
 try:
  if out['finish_reason']!='stop':raise ValueError('Incomplete audit generation')
  review,_,_=complete_review_coverage(parse_object(out['response']),case['segments']);accepted=validate_review(review,case['segments'])
 except (ValueError,KeyError,TypeError) as exc:error=str(exc)
 answers.append({'case_id':case['id'],'segments':case['segments'],'response':out['response'],'accepted_segment_ids':sorted(accepted),
  'status':'blocked' if error else 'audit_approved' if len(accepted)==len(case['segments']) else 'audit_rejected'})
(p/'replayed.json').write_text(json.dumps(answers))
'''
    with tempfile.TemporaryDirectory(prefix='generated_audit_parser_') as temp:
        d=Path(temp)
        for n,raw in values.items():(d/n).write_bytes(raw)
        (d/'diagnostic_spec.json').write_bytes(enc(model_cases(spec)));(d/'recorded_rows.json').write_text(json.dumps(rows))
        subprocess.run([sys.executable,'-c',code],cwd=d,check=True,capture_output=True,text=True,timeout=60)
        return json.loads((d/'replayed.json').read_text())
