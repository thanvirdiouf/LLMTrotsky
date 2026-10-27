"""Five independent source readings; neutral checker remains experimental."""
import json
from pathlib import Path
from candidate_audit import validate_protocol as validate_audit,run as run_candidate
from candidate_base_probe import CANDIDATE,sha
from source_reading import reading_parts,display
from source_audit import source_audit_request
from ghost_chat import complete_review_coverage
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError,parse_object,validate_review,REFUSAL,REVIEW_FAILURE


def validate_protocol(p):
    if p.get('purpose')!='candidate_source_synthesis' or p.get('independent_questions')!=5 or p.get('maximum_displayed_answers')!=5:
        raise ValueError('Expected five independent synthesis cases')
    validate_audit({**p,'purpose':'candidate_neutral_audit'})


def evaluate(jobs,generate,output):
    if len(jobs)!=5 or len({j['case_id'] for j in jobs})!=5:raise ValueError('Expected five distinct independent questions')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[];calls=[]
    with (output/'answers.jsonl').open('x') as answers,(output/'model_calls.jsonl').open('x') as journal:
        for job in jobs:
            question=job['question'];data=json.loads(job['messages'][1]['content'])
            if data['current_user_request']!=question:raise ValueError('Writer request differs')
            trace={'matches':[{'packet':p} for p in data['source_packets']]};packet_ids=[p['id'] for p in data['source_packets']]
            stages=[];segments=[];origins=[];accepted=set();error=None;missing=[];unknown=[]
            def call(request):
                out=generate(request['messages']);row={'case_id':job['case_id'],'stage':request['stage'],'messages':request['messages'],'output':out,'output_sha256':fingerprint(out)}
                calls.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush();stages.append({**request,'output':out})
                if out.get('finish_reason')!='stop':raise EvidenceError('Incomplete or truncated synthesis stage')
                return out['response']
            try:
                draft=call({'stage':'draft','messages':job['messages'],'prompt_sha256':fingerprint(job['messages'])}).strip()
                segments,origins=reading_parts(draft,trace,question)
                request=source_audit_request({'reference_date':data['reference_date']},question,trace,segments)
                raw=call(request);review,missing,unknown=complete_review_coverage(parse_object(raw),segments)
                accepted=validate_review(review,segments)
            except (EvidenceError,KeyError,TypeError) as exc:error=str(exc);accepted=set()
            response=display(segments,accepted,packet_ids)
            if not response:response=REVIEW_FAILURE if error or segments else REFUSAL
            elif len(accepted)<len(segments):response+='\n\nI could not support the rest of the requested answer.'
            row={'case_id':job['case_id'],'question':question,'model':CANDIDATE,'application_mode':'experimental_candidate_source_synthesis',
                'response':response,'response_sha256':fingerprint(response),'segments':segments,'accepted_segment_ids':sorted(accepted),
                'source_context_packet_ids':packet_ids,'quotation_origins':origins,'missing_verdict_ids':missing,'unknown_verdict_ids':unknown,
                'status':'blocked' if error else 'partial' if accepted and len(accepted)<len(segments) else 'supported_by_model_review' if accepted else 'review_rejected' if segments else 'abstained',
                'validation_error':error,'stages':stages,'semantic_guarantee':False,'accepted_claim_ids':[],'memory_updates':[],
                'reference_date':data['reference_date'],'training_steps':0,'adapters_loaded':[]}
            rows.append(row);answers.write(json.dumps(row,ensure_ascii=False)+'\n');answers.flush()
            print(job['case_id'],row['status'],flush=True)
    return {'displayed_answers':len(rows),'completed_evaluation_calls':len(calls),'answers_sha256':sha((output/'answers.jsonl').read_bytes()),'model_calls_sha256':sha((output/'model_calls.jsonl').read_bytes())}


def run(bundle,output):
    import sys
    return run_candidate(bundle,output,experiment=sys.modules[__name__])
