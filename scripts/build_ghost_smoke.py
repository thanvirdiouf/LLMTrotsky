#!/usr/bin/env python3
"""Freeze a ten-answer ghost smoke and a separate 5,000-token memory probe."""
import ast
import argparse
import base64
from copy import deepcopy
import hashlib
import json
import lzma
import re
from pathlib import Path
from grounded_chat import ROOT
from ghost_chat import GhostChat,load_active,initial_state,claims_for_trace,request


def enc(value):return (json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()


def model_cases(source_spec):
    if 'factual_comparison' in source_spec:
        if (source_spec['factual_comparison'].get('arms') not in (['canonical','reading'],['reading','mapped'])
                or source_spec.get('conversation') or len(source_spec['cases'])!=5
                or len({c['id'] for c in source_spec['cases']})!=5
                or any(not isinstance(c['source_packet_ids'],list) or not 1<=len(c['source_packet_ids'])<=2
                    or len(set(c['source_packet_ids']))!=len(c['source_packet_ids']) for c in source_spec['cases'])):
            raise ValueError('Expected five independent matched-source factual cases')
        return {'factual_comparison':{'arms':source_spec['factual_comparison']['arms']},'conversation':[],
            'cases':[{'id':arm+'_'+c['id'],'question':c['question'],'source_packet_ids':c['source_packet_ids'],
                      'comparison_arm':arm,'paired_case_id':c['id']} for arm in source_spec['factual_comparison']['arms'] for c in source_spec['cases']]}
    if 'audit_calibration' in source_spec:
        control=source_spec['audit_calibration']
        if control.get('policy','ghost') not in ('ghost','neutral_source'):
            raise ValueError('Unknown audit policy')
        return {'audit_calibration':{k:control[k] for k in ('packet_ids','policy') if k in control},'conversation':[],
            'cases':[{k:c[k] for k in ('id','question','claim_ids','segments','history') if k in c} for c in source_spec['cases']]}
    if 'comparison' not in source_spec:
        result={'cases':[{'id':c['id'],'question':c['question']} for c in source_spec['cases']],
                'conversation':source_spec['conversation']}
        if source_spec.get('controlled_sources'):
            for case,original in zip(result['cases'],source_spec['cases']):case['source_packet_ids']=original['source_packet_ids']
            result['conversation_source_packet_ids']=source_spec['conversation_source_packet_ids']
        return result
    control=source_spec['comparison']
    if control['arms']!=['direct','checked'] or source_spec['conversation']:
        raise ValueError('Expected a direct/checked five-turn case sequence')
    return {'comparison':{'arms':control['arms'],'packet_ids':control['packet_ids']},
        'conversation':[],
        'cases':[{'id':arm+'_'+c['id'],'question':c['question'],'arm':arm,'conversation_turn':i}
                 for arm in control['arms'] for i,c in enumerate(source_spec['cases'],1)]}


def build(version='ghost_smoke_v1', evaluation_spec=None, constrain_selection=False,source_plan=None):
    if not re.fullmatch(r'ghost_smoke_v[1-9][0-9]*',version):raise ValueError('Invalid smoke version')
    from transformers import AutoTokenizer
    if source_plan:
        from build_ghost_chat import build as build_evidence
        bot=GhostChat(*build_evidence(source_plan=source_plan))
    else:bot=load_active()
    prior=json.loads((ROOT/'evaluation/grounded_chat_v2/protocol.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'training/data_revision_v2/tokenizer_assets',trust_remote_code=False,local_files_only=True)
    if sha(tokenizer.chat_template.encode())!=prior['chat_template_sha256']:raise ValueError('Tokenizer changed')
    source_spec=json.loads((ROOT/(evaluation_spec or 'config/ghost_chat_evaluation.json')).read_text())
    model_spec=model_cases(source_spec)
    if len(model_spec['cases'])+len(model_spec['conversation'])>10:
        raise ValueError('Standing approval permits at most ten answers/thirty evaluation calls per run')
    state=initial_state(bot.collection,bot.ledger)
    question='Please explain the actual sequence more fully, separating his words from your replies.'
    state['active_packet_ids']=['hearing']
    trace=bot.retriever.retrieve(question,state);cards=claims_for_trace(bot.ledger,trace)
    segment=[{'id':'s1','kind':'fact','text':'The occupation question precedes household membership.','claim_ids':['hearing_sequence']}]
    probe=request('check',bot.profile,question,trace,cards,[],state,segment)
    data=json.loads(probe['messages'][1]['content'])
    # Neutral padding only tests memory. This request is never scored or used
    # as conversation history, and its 16-token reply is not an evaluation answer.
    def padded(count):
        messages=deepcopy(probe['messages']);value={**data,'memory_probe_only':'padding '*count}
        messages[1]['content']=json.dumps(value,ensure_ascii=False,separators=(',',':'))
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        return messages,ids
    low,high=0,6000
    while low<high:
        mid=(low+high+1)//2
        if len(padded(mid)[1])<=5000:low=mid
        else:high=mid-1
    messages,ids=padded(low)
    if not 4900<=len(ids)<=5000:raise ValueError('Memory probe length differs')
    probe={'messages':messages,'input_token_ids':ids,'purpose':'unscored memory probe; no factual quality inference'}
    names=['ghost_chat.py','grounded_chat.py','grounded_chat_v2.py','evaluate_ghost_smoke.py',
           'evaluate_grounded_chat_v2.py','kaggle_ghost_smoke.py','kaggle_grounding.py','kbit_preparation.py']
    if 'comparison' in model_spec or 'conversation_source_packet_ids' in model_spec:names.append('evaluate_ghost_comparison.py')
    if 'audit_calibration' in model_spec:names.extend(['evaluate_ghost_comparison.py','evaluate_ghost_audit.py'])
    if model_spec.get('audit_calibration',{}).get('policy')=='neutral_source':names.append('source_audit.py')
    if 'factual_comparison' in model_spec:names.extend(['evaluate_ghost_comparison.py','source_reading.py','mapped_source_reading.py','evaluate_ghost_factual_comparison.py'])
    if constrain_selection:names.append('selection_constraints.py')
    payload={n:(ROOT/'scripts'/n).read_bytes() for n in names}
    payload.update({'profile.json':enc(bot.profile),'collection.json':enc(bot.collection),'claims.json':enc(bot.ledger),
                    'model_cases.json':enc(model_spec),'memory_probe.json':enc(probe)})
    protocol={k:deepcopy(prior[k]) for k in ['model','packages','generation','chat_template_sha256']}
    protocol.update(purpose='present_day_ghost_smoke',answers=len(model_spec['cases'])+len(model_spec['conversation']),
        maximum_model_calls=3*(len(model_spec['cases'])+len(model_spec['conversation'])),memory_probe_calls=1,
        memory_probe_tokens=len(ids),max_input_tokens=5120,training_steps=0,adapters_loaded=[],
        evaluation_spec_sha256=sha(enc(source_spec)),files={n:sha(b) for n,b in payload.items()},
        interpretation_policy='Fictional contemporary opinions are allowed, historical/modern factual premises need sources; no 1940 knowledge cutoff.',
        preserved_files=json.loads((ROOT/'evaluation/grounded_chat_v2/preserved_v1_and_adapters.json').read_text()))
    if 'comparison' in model_spec:
        protocol.update(purpose='ghost_direct_checked_comparison',maximum_model_calls=sum(1 if c['arm']=='direct' else 3 for c in model_spec['cases']))
    if 'audit_calibration' in model_spec:
        protocol.update(purpose='ghost_audit_calibration',maximum_model_calls=len(model_spec['cases']))
    if 'factual_comparison' in model_spec:
        protocol.update(purpose='ghost_factual_comparison',maximum_model_calls=25)
    if constrain_selection:
        from selection_constraints import POLICY
        protocol['selection_constraints']=deepcopy(POLICY)
    if source_plan:protocol['source_plan']={'path':str(source_plan),'sha256':sha((ROOT/source_plan).read_bytes())}
    payload['protocol.json']=enc(protocol)
    cells=[]
    def add(kind,cid,source):
        item={'cell_type':kind,'id':cid,'metadata':{},'source':source.splitlines(True)}
        if kind=='code':
            item.update(execution_count=None,outputs=[])
            if cid!='install':compile(source,cid,'exec')
        cells.append(item)
    intro='# Present-day Trotsky ghost smoke\nFictional present-day persona, no 1940 cutoff. Untouched base, no adapters/training. Separate 5,000-token memory probe followed by ten answers, at most thirty evaluation generation calls. Structural checks and model audits are not semantic proof.\n'
    if 'comparison' in model_spec:
        intro='# Direct/checked present-day ghost comparison\nSame untouched base and fixed available source context. Two separate five-turn conversations with actual displayed history; ten answers, at most twenty evaluation calls plus one memory probe. Direct replies are unchecked research outputs; no training/adapters. Whole-pipeline comparison, not an ablation or independent historical validation.\n'
    if 'audit_calibration' in model_spec:
        intro='# Fictional ghost auditor calibration\nTen fixed candidate test inputs, at most ten checker calls plus one memory probe. Candidates include deliberately false assertions and are never displayed as chatbot answers or trusted memory. Expected decisions are withheld from the model. Untouched base, no training/adapters/paid compute. Development diagnostic, not independent validation.\n'
    if 'factual_comparison' in model_spec:
        intro='# Matched factual source-reading comparison\nFive independent questions in two explicitly named factual pipelines, ten displayed replies total, at most25 calls plus one unscored memory probe. Same pinned base/full source packets; no training/adapters/paid compute. Generated readings remain fallible and never become verified claim memory.\n'
    add('markdown','intro',intro)
    add('code','install','%pip install -q '+' '.join(f'{k}=={v}' for k,v in protocol['packages'].items() if k!='torch')+'\n')
    packed={n:{'sha256':sha(b),'data':base64.b85encode(lzma.compress(b)).decode()} for n,b in payload.items()}
    add('code','unpack','import base64,hashlib,lzma,sys\nfrom pathlib import Path\nbundle=Path("/kaggle/working/ghost_smoke_bundle")\nbundle.mkdir(exist_ok=True)\npacked='+repr(packed)+'\nfor name,value in packed.items():\n    raw=lzma.decompress(base64.b85decode(value["data"]))\n    assert hashlib.sha256(raw).hexdigest()==value["sha256"]\n    (bundle/name).write_bytes(raw)\nsys.path.insert(0,str(bundle))\n')
    add('code','run','from kaggle_ghost_smoke import run\nrun(bundle,"/kaggle/working/ghost_smoke_results")\n')
    notebook=enc({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'cells':cells})
    metadata=json.loads((ROOT/'training/kaggle_cli/grounded_chat_v1/kernel-metadata.json').read_text())
    metadata.update(id_no=140792344,code_file='kaggle_ghost_smoke.ipynb')
    lock={'notebook_sha256':sha(notebook),'metadata_sha256':sha(enc(metadata)),'payload_sha256':{n:sha(b) for n,b in payload.items()}}
    out=ROOT/'evaluation'/version;out.mkdir(parents=True,exist_ok=True)
    for name,raw in {'protocol.json':enc(protocol),'evaluation_spec.json':enc(source_spec),'bundle_lock.json':enc(lock)}.items():
        if (out/name).exists() and (out/name).read_bytes()!=raw:raise ValueError('Frozen ghost smoke differs: '+name)
        (out/name).write_bytes(raw)
    stage=ROOT/'training/kaggle_cli'/version;stage.mkdir(parents=True,exist_ok=True)
    (stage/'kaggle_ghost_smoke.ipynb').write_bytes(notebook);(stage/'kernel-metadata.json').write_bytes(enc(metadata))
    (ROOT/'notebooks/kaggle_ghost_smoke.ipynb').write_bytes(notebook)
    print(json.dumps({'stage':str(stage),'answers':protocol['answers'],'maximum_evaluation_calls':protocol['maximum_model_calls'],
                      'separate_memory_probe_calls':1,'memory_probe_input_tokens':len(ids),'max_input_tokens':5120}))
    return payload,protocol


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',default='ghost_smoke_v1')
    parser.add_argument('--evaluation-spec',type=Path,help='Explicit new case set; default preserves the inspected regressions')
    parser.add_argument('--constrain-selection',action='store_true',help='Experimental finite selection JSON decoding; no extra inference call')
    parser.add_argument('--source-plan',type=Path,help='Freeze an explicit evidence plan without changing the active pointer')
    args=parser.parse_args()
    build(args.version,args.evaluation_spec,args.constrain_selection,args.source_plan)
