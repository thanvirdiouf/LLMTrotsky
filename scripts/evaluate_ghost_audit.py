"""Bounded reviewer calibration; candidates are test inputs, never chat replies."""
from copy import deepcopy
import json
try:
    from .ghost_chat import GhostChat,initial_state,claims_for_trace,request,parse_object,complete_review_coverage
    from .grounded_chat import MODEL,fingerprint
    from .grounded_chat_v2 import validate_review
    from .evaluate_ghost_comparison import FixedRetriever
    from .evaluate_ghost_smoke import evaluate as journal_evaluate
except ImportError:
    from ghost_chat import GhostChat,initial_state,claims_for_trace,request,parse_object,complete_review_coverage
    from grounded_chat import MODEL,fingerprint
    from grounded_chat_v2 import validate_review
    from evaluate_ghost_comparison import FixedRetriever
    from evaluate_ghost_smoke import evaluate as journal_evaluate


def evaluate(profile,collection,ledger,spec,generate,output,identity):
    if spec.get('conversation') or not 1<=len(spec['cases'])<=10:raise ValueError('Expected at most ten independent audit cases')
    policy=spec['audit_calibration'].get('policy','ghost')
    if policy not in ('ghost','neutral_source'):raise ValueError('Unknown audit policy')
    if len({c['id'] for c in spec['cases']})!=len(spec['cases']):raise ValueError('Duplicate audit case ID')
    class AuditBot(GhostChat):
        def __init__(self,*args):
            super().__init__(*args);self.cases=iter(spec['cases'])
            self.retriever=FixedRetriever(self,spec['audit_calibration']['packet_ids'])

        def answer(self,question,generate,*,state=None,history=()):
            case=next(self.cases)
            if case['question']!=question:raise ValueError('Audit case order differs')
            state=deepcopy(state or initial_state(self.collection,self.ledger))
            trace=self.retriever.retrieve(question,state);available=claims_for_trace(self.ledger,trace)
            selected={cid:available[cid] for cid in case['claim_ids']}
            # Deliberately include false test candidates without canonicalizing
            # their text. Only the judge JSON is displayed; no claims are trusted.
            segments=deepcopy(case['segments'])
            temp={**state,'_answer_mode':'view' if any(s['kind']=='interpretation' for s in segments) else 'facts','_detail':'normal'}
            if policy=='neutral_source':
                try:
                    from .source_audit import source_audit_request
                except ImportError:
                    from source_audit import source_audit_request
                req=source_audit_request(self.profile,question,trace,segments)
            else:
                req=request('check',self.profile,question,trace,selected,case.get('history',[]),temp,segments)
            out=generate(req['messages']);error=None;accepted=set();missing=[];unknown=[]
            try:
                if out.get('finish_reason')!='stop':raise ValueError('Incomplete audit generation')
                review,missing,unknown=complete_review_coverage(parse_object(out['response']),segments)
                accepted=validate_review(review,segments)
            except (ValueError,KeyError,TypeError) as exc:error=str(exc)
            response=out['response']
            return {'mode':'reviewer_calibration','display_label':self.profile['display_label'],'reference_date':self.profile['reference_date'],
                'historical_knowledge_cutoff':None,'response':response,'response_sha256':fingerprint(response),
                'state_before':state,'state_after':{**state,'turns':state['turns']+1},'retrieval':trace,'segments':segments,
                'accepted_segment_ids':sorted(accepted),'selected_claim_ids':list(selected),'stages':[{**req,'output':out}],
                'audit_candidate_display':json.loads(req['messages'][1]['content']).get('candidate_display','\n\n'.join(s['text'] for s in segments)),
                'missing_verdict_ids':missing,'unknown_verdict_ids':unknown,
                'status':'blocked' if error else 'audit_approved' if len(accepted)==len(segments) else 'audit_rejected',
                'validation_error':error,'semantic_guarantee':False,'model':MODEL,'training_steps':0,'adapters_loaded':[]}
    return journal_evaluate(profile,collection,ledger,spec,generate,output,identity,bot_factory=AuditBot)
