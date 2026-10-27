"""Five controlled-source ID selections, rendered from immutable reviewed records."""
import json
from pathlib import Path
from candidate_audit import run as run_candidate,validate_protocol as validate_common
from candidate_base_probe import CANDIDATE,sha
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError
from reviewed_source_display import records_for_packets,select_ids,render_selection,MAX_SELECTED


def validate_protocol(p):
    if (p.get('purpose')!='candidate_reviewed_selection' or p.get('evaluation_calls')!=5
            or p.get('independent_questions')!=5 or p.get('maximum_displayed_answers')!=5
            or p.get('maximum_selected_records')!=MAX_SELECTED or p.get('generated_factual_text_used') is not False
            or p.get('model_audit_performed') is not False):raise ValueError('Reviewed selector scope differs')
    validate_common({**p,'purpose':'candidate_neutral_audit','evaluation_calls':10})


def evaluate(jobs,generate,output,collection,ledger):
    if len(jobs)!=5 or len({j['case_id'] for j in jobs})!=5:raise ValueError('Expected five independent selector cases')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[];calls=[]
    with (output/'answers.jsonl').open('x') as answers,(output/'model_calls.jsonl').open('x') as journal:
        for job in jobs:
            records=records_for_packets(collection,ledger,job['source_packet_ids'])
            if {i:r['record_fingerprint'] for i,r in records.items()}!=job['available_record_fingerprints']:raise ValueError('Reviewed menu differs from frozen inputs')
            out=generate(job['messages']);call={'case_id':job['case_id'],'stage':'select','messages':job['messages'],'output':out,'output_sha256':fingerprint(out)}
            calls.append(call);journal.write(json.dumps(call,ensure_ascii=False)+'\n');journal.flush();error=None
            try:
                if out.get('finish_reason')!='stop':raise EvidenceError('Incomplete or truncated source-record selection')
                ids=select_ids(out['response'],records);display=render_selection(ids,records)
            except (ValueError,KeyError,TypeError) as exc:
                error=str(exc);display=render_selection([],records)
                display['response']='I could not select reviewed source summaries in the required format. The passages may contain relevant evidence.'
                display['response_fingerprint']=fingerprint(display['response'])
            row={**display,'case_id':job['case_id'],'question':job['question'],'model':CANDIDATE,
                'application_mode':'experimental_reviewed_source_selection','source_context_packet_ids':job['source_packet_ids'],
                'status':'blocked' if error else 'reviewed_summaries_selected' if display['selected_claim_ids'] else 'no_summaries_selected',
                'validation_error':error,'reference_date':collection['reference_date'],'training_steps':0,'adapters_loaded':[]}
            rows.append(row);answers.write(json.dumps(row,ensure_ascii=False)+'\n');answers.flush()
            print(job['case_id'],row['status'],len(row['selected_claim_ids']),flush=True)
    return {'displayed_answers':len(rows),'completed_evaluation_calls':len(calls),
        'answers_sha256':sha((output/'answers.jsonl').read_bytes()),'model_calls_sha256':sha((output/'model_calls.jsonl').read_bytes()),
        'generated_factual_text_used':False,'model_audit_performed':False}


class Experiment:
    def __init__(self,bundle):self.bundle=Path(bundle)
    def validate_protocol(self,p):validate_protocol(p)
    def evaluate(self,jobs,generate,output):
        return evaluate(jobs,generate,output,*[json.loads((self.bundle/n).read_text()) for n in ('collection.json','ledger.json')])


def run(bundle,output):return run_candidate(bundle,output,experiment=Experiment(bundle))
