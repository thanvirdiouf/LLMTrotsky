"""Freeze an opt-in ten-reply review run; no fixed-transfer benchmark or training."""
import argparse,ast,base64,json,lzma,re
from pathlib import Path
from build_ghost_candidate_probe import ROOT,enc,sha
from candidate_review_chat import validate_protocol
from review_chat import load_snapshot,prepare_turn
from grounded_request_boundary import ConversationState


def build(version='ghost_review_chat_v1',root=ROOT,tokenizer=None):
    if not re.fullmatch(r'ghost_review_chat_v[1-9][0-9]*',version):raise ValueError('Invalid review version')
    design_raw=(root/'config/ghost_review_workbench_v1.json').read_bytes();design=json.loads(design_raw)
    if len(design['conversations'])!=5 or sum(len(c['questions']) for c in design['conversations'])!=10:raise ValueError('Conversation scope differs')
    source=root/design['source_snapshot'];profile,collection,ledger=load_snapshot(source)
    old=json.loads((root/'evaluation/ghost_candidate_audit_v2/protocol.json').read_text())
    assets=root/'evaluation/ghost_reviewed_target_inputs_v2/tokenizer_assets'
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(assets,local_files_only=True,trust_remote_code=False)
    if sha(tokenizer.chat_template.encode())!=old['chat_template_sha256']:raise ValueError('Candidate template differs')
    jobs=[]
    for conversation in design['conversations']:
        state=ConversationState(tuple(p['id'] for p in collection['packets']))
        first=prepare_turn(profile,collection,ledger,state,conversation['questions'][0],tokenizer=tokenizer,
            reflection_allowed=conversation['reflection_allowed'],today=design['current_date'])
        jobs.append({'case_id':conversation['id'],'title':conversation['title'],'questions':conversation['questions'],
            'reflection_allowed':conversation['reflection_allowed'],'messages':first['messages'],'input_token_ids':first['input_token_ids']})
    p={k:old[k] for k in ('model','packages','generation','gpu_ids','placement_policy','attention_policy','tokenizer_files','chat_template_sha256',
        'per_gpu_weight_budget_bytes','weight_file_bytes','preserved_files','runtime_timeout_seconds','installation_timeout_seconds')}
    p.update(purpose='candidate_conversation_review',conversation_count=5,maximum_displayed_answers=10,evaluation_calls=10,memory_probe_calls=1,
        max_input_tokens=5120,max_output_tokens=1024,memory_probe_output_tokens=16,enable_thinking=False,training_steps=0,adapters_loaded=[],
        active_model_changed=False,model_audit_performed=False,human_review_only=True,fixed_transfer_questions_run=False,
        design_sha256=sha(design_raw),source_manifest_sha256=sha((source/'build_manifest.json').read_bytes()),
        source_review_sha256=sha((source/'SOURCE_REVIEW.json').read_bytes()),
        quality_policy='Experimental generated conversations; citation checks are mechanical, all factual support and voice require human/source review. No activation or trusted-memory promotion.')
    names=('candidate_audit.py','candidate_base_probe.py','candidate_efficient_attention.py','candidate_review_chat.py',
        'review_chat.py','grounded_chat.py','grounded_chat_v2.py','grounded_request_boundary.py')
    payload={n:(root/'scripts'/n).read_bytes() for n in names}
    from verify_reviewed_selection import payloads as control_payloads
    control=control_payloads('ghost_reviewed_selection_v1',root)
    for n in ('candidate_audit.py','candidate_base_probe.py','candidate_efficient_attention.py','grounded_chat.py','grounded_chat_v2.py'):
        if payload[n]!=control[n]:raise ValueError('Shared control changed: '+n)
    payload.update({n:(source/n).read_bytes() for n in ('profile.json','collection.json','ledger.json')})
    payload.update({'audit_inputs.json':enc(jobs),'review_design.json':design_raw,
        'memory_probe.json':(root/'evaluation/ghost_reviewed_target_inputs_v2/memory_probe.json').read_bytes()})
    p['files']={n:sha(raw) for n,raw in payload.items()};validate_protocol(p);payload['protocol.json']=enc(p)
    packed={n:base64.b85encode(lzma.compress(raw)).decode() for n,raw in payload.items()}
    install='import subprocess,sys\nsubprocess.run([sys.executable,"-m","pip","install","-q",'+','.join(repr(k+'=='+v) for k,v in p['packages'].items() if k!='torch')+'],check=True,timeout=600)\n'
    run='import base64,lzma,sys\nfrom pathlib import Path\npacked='+repr(packed)+'\nbundle=Path("/kaggle/working/ghost_review_chat_bundle")\nbundle.mkdir(exist_ok=True)\nfor name,value in packed.items():\n    (bundle/name).write_bytes(lzma.decompress(base64.b85decode(value)))\nsys.path.insert(0,str(bundle))\nfrom candidate_review_chat import run\nrun(bundle,"/kaggle/working/ghost_review_chat_results")\n'
    for text in (install,run):ast.parse(text)
    intro='Experimental present-day fictional Trotsky conversations for human review. Five conversations, ten replies maximum/ten evaluation calls plus one5000/16probe; pinned27B/twofreeT4s/private existing notebook. No training/adapters/paid compute/active replacement/self-audit. Real retrieval, no oracle source IDs. Setup600s/workload2700s; unvalidated factual support/voice. New end-to-end prototype authorized by user; not another familiar selector-prompt retry.\n'
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'cells':[
        {'cell_type':'markdown','id':'scope','metadata':{},'source':[intro]},
        *[{'cell_type':'code','id':name,'metadata':{},'source':text.splitlines(True),'execution_count':None,'outputs':[]} for name,text in [('install',install),('run',run)]]]}
    metadata=json.loads((root/'training/kaggle_cli/ghost_candidate_audit_v2/kernel-metadata.json').read_text());metadata['code_file']='kaggle_review_chat.ipynb'
    if metadata['is_private'] is not True or metadata['id_no']!=140792344:raise ValueError('Private target differs')
    stage=root/'training/kaggle_cli'/version;record=root/'evaluation'/version
    raw=enc(notebook);meta_raw=enc(metadata);lock={'notebook_sha256':sha(raw),'metadata_sha256':sha(meta_raw),'payload_sha256':{n:sha(raw) for n,raw in payload.items()}}
    files={stage/metadata['code_file']:raw,stage/'kernel-metadata.json':meta_raw,record/'protocol.json':payload['protocol.json'],record/'bundle_lock.json':enc(lock)}
    for name,digest in p['tokenizer_files'].items():
        raw=(assets/name).read_bytes()
        if sha(raw)!=digest:raise ValueError('Tokenizer asset changed')
        files[record/'tokenizer_assets'/name]=raw
    for path,raw in files.items():
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Frozen review run differs; use a new version')
    for path,raw in files.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return {'stage':str(stage),'initial_input_tokens':[len(j['input_token_ids']) for j in jobs]}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',default='ghost_review_chat_v1');print(json.dumps(build(p.parse_args().version),indent=2))
