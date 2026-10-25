#!/usr/bin/env python3
"""Verify controlled lower-LR artifacts and export blank blinded review rows.

No inference, training, historical scoring or mutation of frozen data occurs.
The downloaded delivery archive is an independent local copy used to detect
post-extraction edits; hashes are integrity evidence, not execution attestation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

if __package__:
    from . import lr_test_protocol as protocol_tools
    from . import kaggle_checkpoint_compare as journal_tools
else:
    import lr_test_protocol as protocol_tools
    import kaggle_checkpoint_compare as journal_tools

PROJECT=Path(__file__).resolve().parents[1]
CONDITIONS=protocol_tools.CONDITIONS
SCORE_FIELDS=('chronology','correction','evidence_calibration','answer_detail','critical_failure','review_notes')
REQUIRED_CHECKS=('all_50_masks_and_complete_answers','exact_text_adapters_and_frozen_base',
 'longest_example_finite_and_fits','resumable_checkpoint_round_trip',
 'all_50_finite_verified_optimizer_steps','all_examples_seen_exactly_once',
 'fresh_reload_exact_adapter_tensors','fresh_reload_generation_matches',
 'evaluation_base_untouched','original_step50_exact_tensors','revised_eval_exact_tensors',
 'revised_eval_fresh_base_reload','matched_fresh_and_regression_evaluation_complete','matched_initial_adapter_bytes')


def digest(payload):return hashlib.sha256(payload).hexdigest()
def fingerprint(value):return protocol_tools.fingerprint(value)
def _json(payload):
    def reject(value):raise ValueError('Non-finite JSON: '+value)
    return json.loads(payload,parse_constant=reject)
def _jsonl(payload):
    result=[_json(line) for line in payload.splitlines()]
    if not all(isinstance(row,dict) for row in result):raise ValueError('Expected JSONL objects')
    return result
def _file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def _safe_name(name):
    p=Path(name)
    return not p.is_absolute() and '..' not in p.parts and bool(name) and '\\' not in name


def verify_delivery_archive(run_dir,archive_path=None):
    """Compare every delivery entry to its extracted file; never extract paths."""
    run_dir=Path(run_dir)
    if archive_path is None:
        candidates=sorted(run_dir.parent.rglob('lr_test_delivery.zip'))
        if len(candidates)!=1:raise ValueError('Expected one preserved lr_test_delivery.zip; pass --archive explicitly')
        archive_path=candidates[0]
    archive_path=Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        entries=archive.infolist();names=[entry.filename for entry in entries]
        if len(names)!=len(set(names)):raise ValueError('Duplicate delivery archive entry')
        for entry in entries:
            if not _safe_name(entry.filename) or entry.is_dir():raise ValueError('Unsafe/unexpected delivery archive path')
            path=run_dir/entry.filename
            if path.is_symlink() or not path.is_file():raise ValueError('Missing/symlink delivery file: '+entry.filename)
            h=hashlib.sha256()
            with archive.open(entry) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=_file_hash(path):raise ValueError('Delivery archive mismatch: '+entry.filename)
        required={'run_manifest.json','comparison_manifest.json','protocol.json','training_lock.json',
          'model_inputs.jsonl','training_order.json','training_metrics.jsonl','tokenization_report.json',
          'training_summary.json','trained_probe.json','reloaded_probe.json','evaluation_summary.json',
          'final_adapter/adapter_config.json','final_adapter/adapter_model.safetensors',
          'inputs/train_slice.jsonl','inputs/pilot_run_manifest.json',
          *[f'{condition}_responses.jsonl' for condition in CONDITIONS]}
        if not required<=set(names):raise ValueError('Incomplete delivery archive')
    return {'archive_sha256':_file_hash(archive_path),'files_verified':len(names)}


def _checkpoint_manifest(path):
    path=Path(path);raw=(path/'checkpoint_manifest.json').read_bytes();m=_json(raw)
    marker=_json((path/'COMPLETE').read_bytes())
    if marker.get('manifest_sha256')!=digest(raw):raise ValueError('Checkpoint completion marker differs')
    return m,digest(raw)


def _local_checkpoint_diagnostics(run_dir,project,protocol,identity):
    control_paths=list((project/'evaluation/runs/kaggle_data_revision_v2_v1/download').rglob('checkpoint-000050/checkpoint_manifest.json'))
    if len(control_paths)!=1:raise ValueError('Missing saved revised control checkpoint manifest')
    control,control_hash=_checkpoint_manifest(control_paths[0].parent)
    if (control_hash!=protocol['original_checkpoint_manifest_sha256']
        or control['identity']!=protocol['control_identity'] or control['completed_steps']!=50
        or control['files']['adapter/adapter_model.safetensors']!=protocol['control_adapter_sha256']):
        raise ValueError('Saved revised lr1e4 comparator provenance differs')
    new_paths=list(run_dir.parent.rglob('checkpoint-000050/checkpoint_manifest.json'))
    zeros=list(run_dir.parent.rglob('checkpoint-000000/checkpoint_manifest.json'))
    if len(new_paths)!=1 or len(zeros)!=1:raise ValueError('Download new step0/step50 checkpoint manifests and COMPLETE')
    new,new_hash=_checkpoint_manifest(new_paths[0].parent);zero,zero_hash=_checkpoint_manifest(zeros[0].parent)
    if (new['identity']!=identity or new['completed_steps']!=50 or new['order_ids']!=protocol['training_order']['example_ids']
        or new['files']['adapter/adapter_model.safetensors']!=_file_hash(run_dir/'final_adapter/adapter_model.safetensors')):
        raise ValueError('New checkpoint identity/order/adapter differs')
    if (zero['identity']!=identity or zero['completed_steps']!=0
        or zero['files']['adapter/adapter_model.safetensors']!=protocol['control_initial_adapter_sha256']):
        raise ValueError('Initial adapter differs from saved lr1e4 control')
    return dict(saved_revised_control_verified=True,new_step50_manifest_verified=True,
                matched_initial_adapter_bytes=True,control_adapter_sha256=protocol['control_adapter_sha256'],
                new_adapter_sha256=new['files']['adapter/adapter_model.safetensors'],
                new_checkpoint_manifest_sha256=new_hash,initial_checkpoint_manifest_sha256=zero_hash)


def validate_run(run_dir,*,project=PROJECT,archive_path=None):
    run_dir,project=Path(run_dir),Path(project)
    archive=verify_delivery_archive(run_dir,archive_path)
    data=project/'datasets/revision_v2';training=project/'training/lr_1e5';fresh=project/'evaluation/fresh_holdout_v3'
    protocol_bytes=(training/'protocol.json').read_bytes();protocol=_json(protocol_bytes)
    if (run_dir/'protocol.json').read_bytes()!=protocol_bytes:raise ValueError('Downloaded training protocol differs')
    version=_json((data/'version_manifest.json').read_bytes())
    for name,expected in version['files'].items():
        if not _safe_name(name) or digest((data/name).read_bytes())!=expected:raise ValueError('Frozen data version differs: '+name)
    if digest((data/'version_manifest.json').read_bytes())!=protocol['training']['version_manifest_sha256']:raise ValueError('Version manifest identity differs')
    exposure_bytes=(project/'training/data_revision_v2/token_mask_preflight.json').read_bytes();exposure=_json(exposure_bytes)
    if digest(exposure_bytes)!=protocol['exposure_report_sha256']:raise ValueError('Frozen token exposure differs')
    fresh_bytes=(fresh/'protocol.json').read_bytes();fresh_protocol=_json(fresh_bytes)
    if digest(fresh_bytes)!=protocol['evaluation']['source_protocol_sha256']:raise ValueError('Frozen evaluation protocol differs')
    for name,expected in fresh_protocol['files'].items():
        if not _safe_name(name) or digest((fresh/name).read_bytes())!=expected:raise ValueError('Frozen evaluation file differs: '+name)
    if digest((fresh/'specifications.json').read_bytes())!=fresh_protocol['specifications_sha256'] or digest((fresh/'source_review.md').read_bytes())!=fresh_protocol['source_review_sha256']:
        raise ValueError('Frozen evaluation specification/source review differs')
    train_bytes=(data/'train_slice.jsonl').read_bytes();eval_bytes=(training/'model_inputs.jsonl').read_bytes();lock_bytes=(data/'training_lock.json').read_bytes()
    if (run_dir/'inputs/train_slice.jsonl').read_bytes()!=train_bytes or (run_dir/'training_lock.json').read_bytes()!=lock_bytes or (run_dir/'model_inputs.jsonl').read_bytes()!=eval_bytes:
        raise ValueError('Downloaded training/evaluation payload differs')
    pilot_bytes=(project/'training/runs/kaggle_361474126/run_manifest.json').read_bytes()
    if digest(pilot_bytes)!=protocol['pilot_run_manifest_sha256'] or (run_dir/'inputs/pilot_run_manifest.json').read_bytes()!=pilot_bytes:
        raise ValueError('Original pilot manifest bytes differ')
    control_bytes=(project/'evaluation/runs/kaggle_data_revision_v2_v1/results/run_manifest.json').read_bytes()
    if ((run_dir/'inputs/control_run_manifest.json').read_bytes()!=control_bytes
        or digest(control_bytes)!=protocol['control_run_manifest_sha256']):raise ValueError('Control manifest differs')
    expected_eval=(fresh/'model_inputs.jsonl').read_bytes()+(project/'evaluation/fresh_holdout_v2/model_inputs.jsonl').read_bytes()
    if eval_bytes!=expected_eval:raise ValueError('Combined fresh/regression projection differs')
    if digest((project/'evaluation/fresh_holdout_v2/protocol.json').read_bytes())!=protocol['evaluation']['regression_protocol_sha256']:
        raise ValueError('Regression protocol changed')
    regression=project/'evaluation/fresh_holdout_v2'
    regression_protocol=_json((regression/'protocol.json').read_bytes())
    for name,expected_hash in regression_protocol['files'].items():
        if not _safe_name(name) or digest((regression/name).read_bytes())!=expected_hash:
            raise ValueError('Frozen regression artifact differs: '+name)
    for name,key in [('specifications.json','specifications_sha256'),('source_review.md','source_review_sha256')]:
        if digest((regression/name).read_bytes())!=regression_protocol[key]:raise ValueError('Regression source review differs')
    train_rows,inputs=protocol_tools.validate_payloads(train_bytes,eval_bytes,_json(lock_bytes),protocol,_json(pilot_bytes))
    if [r['example_id'] for r in train_rows]!=[r['example_id'] for r in _jsonl(train_bytes)]:raise ValueError('Unexpected training projection')
    cases=_jsonl((fresh/'cases.jsonl').read_bytes())+_jsonl((project/'evaluation/fresh_holdout_v2/cases.jsonl').read_bytes())
    if len(cases)!=48 or [c['case_id'] for c in cases]!=[r['case_id'] for r in inputs]:raise ValueError('Reviewer case coverage differs')
    for case,inp in zip(cases,inputs):
        if any(case.get(k)!=v for k,v in inp.items()) or digest(case['reference_evidence'].encode())!=case['evidence_sha256']:
            raise ValueError('Reviewer prompt/evidence differs')
        if case.get('control_type') not in {'positive','negative','conditional'} or not case.get('reference_checks'):
            raise ValueError('Missing frozen control/checklist')
    manifest=_json((run_dir/'run_manifest.json').read_bytes());identity=manifest.get('identity');comparison=_json((run_dir/'comparison_manifest.json').read_bytes())
    if not isinstance(identity,dict):raise ValueError('Missing run identity')
    identity_sha=fingerprint(identity)
    if (manifest.get('status')!='complete' or manifest.get('stage')!='complete' or manifest.get('completed_steps')!=50
        or manifest.get('identity_sha256')!=identity_sha or manifest.get('resume_checkpoint') is not None
        or any(manifest.get('checks',{}).get(k) is not True for k in REQUIRED_CHECKS)
        or not all(v is True for v in manifest.get('checks',{}).values())):
        raise ValueError('Incomplete run, changed identity/checks, or nonfresh training adapter')
    expected={'settings':protocol['settings'],'packages':protocol['packages'],'model_id':protocol['settings']['model_id'],
      'revision':protocol['settings']['revision'],'dataset_sha256':digest(train_bytes),'pilot_lock_sha256':digest(lock_bytes),
      'protocol_sha256':digest(protocol_bytes),'evaluation_inputs_sha256':digest(eval_bytes),'chat_template_sha256':protocol['chat_template_sha256'],
      'order_sha256':protocol['training_order']['ordered_example_ids_sha256'],'helpers':protocol['frozen_helpers'],
      'gpu':protocol['pilot_identity']['gpu'],'dtype':'torch.float16','gpu_ids_used':[0],'purpose':'controlled_lr_1e5_50_updates'}
    if any(identity.get(k)!=v for k,v in expected.items()):raise ValueError('Run settings/model/packages/data/helpers differ from frozen intervention')
    for name,expected_hash in protocol['frozen_helpers'].items():
        if digest((run_dir/'inputs'/name).read_bytes())!=expected_hash:raise ValueError('Archived helper differs: '+name)
    order=_json((run_dir/'training_order.json').read_bytes())
    ids=protocol['training_order']['example_ids']
    if order.get('example_ids')!=ids or order.get('sha256')!=fingerprint(ids):raise ValueError('Frozen optimizer order differs')
    metrics=_jsonl((run_dir/'training_metrics.jsonl').read_bytes());protocol_tools.verify_metrics(metrics,ids)
    by_exposure={r['example_id']:r for r in exposure['ordered_examples']}
    for row in metrics:
        ref=by_exposure[row['example_id']]
        if row['step']!=ref['update'] or row['input_tokens']!=ref['input_tokens'] or row['supervised_tokens']!=ref['supervised_tokens']:
            raise ValueError('Actual optimizer token exposure differs from CPU freeze')
    token_report=_json((run_dir/'tokenization_report.json').read_bytes())
    if (token_report.get('total')!=50 or token_report.get('retained')!=50 or token_report.get('dropped')!=0
        or token_report.get('supervised_tokens')!=exposure['slice_supervised_tokens']
        or {r['example_id']:(r['token_count'],r['supervised_tokens']) for r in token_report.get('examples',[])}!={r['example_id']:(r['input_tokens'],r['supervised_tokens']) for r in exposure['ordered_examples']}):
        raise ValueError('GPU tokenization report differs from frozen CPU masks/exposure')
    training_summary=_json((run_dir/'training_summary.json').read_bytes())
    if training_summary.get('completed_optimizer_steps')!=50 or training_summary.get('unique_training_examples')!=50 or not training_summary.get('session_changed_adapter_tensors') or training_summary.get('quality_improvement_established') is not False:
        raise ValueError('Training summary does not establish fresh 50-update adapter')
    config=_json((run_dir/'final_adapter/adapter_config.json').read_bytes())
    for field,key in [('base_model_name_or_path','model_id'),('revision','revision'),('r','rank'),('lora_alpha','lora_alpha'),('lora_dropout','lora_dropout')]:
        if config.get(field)!=protocol['settings'][key]:raise ValueError('Final adapter config differs: '+field)
    probes=[_json((run_dir/name).read_bytes()) for name in ['trained_probe.json','reloaded_probe.json']]
    if probes[0].get('token_ids')!=probes[1].get('token_ids') or probes[0].get('input_sha256')!=probes[1].get('input_sha256') or not probes[0].get('token_ids'):
        raise ValueError('Fresh reload probe tokens/input differ')
    generation=comparison.get('generation')
    if (comparison.get('identity_sha256')!=identity_sha or comparison.get('conditions')!=CONDITIONS
        or comparison.get('evaluation_inputs_sha256')!=digest(eval_bytes)
        or comparison.get('evaluation_protocol_sha256')!=protocol['evaluation']['source_protocol_sha256']
        or comparison.get('enable_thinking') is not False or comparison.get('fresh_base_before_any_evaluation_adapter') is not True
        or comparison.get('original_checkpoint_manifest_sha256')!=protocol['original_checkpoint_manifest_sha256']
        or not isinstance(generation,dict) or comparison.get('generation_sha256')!=fingerprint(generation)
        or any(type(generation.get(k)) is not type(v) or generation.get(k)!=v for k,v in protocol['evaluation']['generation'].items())):
        raise ValueError('Comparison generation/provenance differs')
    baseline=_jsonl((run_dir/'fresh_base_responses.jsonl').read_bytes())
    if [r.get('case_id') for r in baseline]!=[r['case_id'] for r in inputs]:raise ValueError('Incomplete fresh baseline')
    prompt_tokens={r['case_id']:r.get('input_token_ids') for r in baseline}
    if (any(not isinstance(ts,list) or not ts or len(ts)>8192 or any(type(t) is not int or t<0 for t in ts) for ts in prompt_tokens.values())
        or identity.get('evaluation_prompt_token_ids_sha256')!=fingerprint(prompt_tokens)):
        raise ValueError('Evaluation input token identity differs')
    results={}
    for condition in CONDITIONS:
        values=journal_tools.read_evaluation_journal(run_dir/f'{condition}_responses.jsonl',identity_sha,condition,inputs,generation,prompt_tokens)
        if len(values)!=48:raise ValueError('Incomplete response condition: '+condition)
        for row,inp in zip(values,inputs):
            if row.get('category')!=inp['category'] or row.get('split')!=inp['split'] or generation['eos_token_id'] in row['token_ids'][:-1]:raise ValueError('Response category/split/interior EOS differs')
        results[condition]=values
    evaluation_summary=_json((run_dir/'evaluation_summary.json').read_bytes())
    if (evaluation_summary.get('complete') is not True or evaluation_summary.get('expected_per_condition')!=48
        or evaluation_summary.get('completed_per_condition')!=dict.fromkeys(CONDITIONS,48)
        or evaluation_summary.get('truncations_by_condition')!={c:sum(r['finish_reason']=='length' for r in rs) for c,rs in results.items()}
        or evaluation_summary.get('quality_scores_assigned') is not False):raise ValueError('Evaluation summary differs')
    checkpoint=_local_checkpoint_diagnostics(run_dir,project,protocol,identity)
    sources={'run/'+name:run_dir/name for name in ['run_manifest.json','comparison_manifest.json','protocol.json','training_lock.json','model_inputs.jsonl','training_order.json','training_metrics.jsonl','tokenization_report.json','training_summary.json','evaluation_summary.json',*[f'{c}_responses.jsonl' for c in CONDITIONS]]}
    sources.update({'fresh/'+name:fresh/name for name in ['protocol.json','cases.jsonl','model_inputs.jsonl','specifications.json','source_review.md']})
    sources.update({'regression/'+name:regression/name for name in ['protocol.json','cases.jsonl','model_inputs.jsonl','specifications.json','source_review.md']})
    verification={'archive':archive,'checkpoint_diagnostics':checkpoint,'actual_updates':50,'responses':144,
                  'identity_sha256':identity_sha,'no_quality_scores_assigned':True}
    return cases,results,manifest,sources,verification


def _lines(values):return ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in values)

def export_review(run_dir,*,project=PROJECT,output_dir=None,seed=20261024,shards=3,archive_path=None):
    run_dir,project=Path(run_dir),Path(project);target=Path(output_dir) if output_dir is not None else run_dir/'review'
    if target.exists():raise FileExistsError('Review directory already exists; preserve scores and use a new directory')
    if type(seed) is not int or type(shards) is not int or not 1<=shards<=48:raise ValueError('Invalid review seed/shard count')
    cases,results,manifest,source_paths,verification=validate_run(run_dir,project=project,archive_path=archive_path)
    keyed_results={c:{r['case_id']:r for r in rs} for c,rs in results.items()}
    ordered=sorted(cases,key=lambda r:digest(f'{seed}\0case\0{r["case_id"]}'.encode()))
    shard_for={r['case_id']:min(i*shards//48,shards-1)+1 for i,r in enumerate(ordered)}
    shuffled=sorted((digest(f'{seed}\0answer\0{case["case_id"]}\0{condition}'.encode()),case,condition) for case in cases for condition in CONDITIONS)
    reviews=[];keys=[]
    for number,(_,case,condition) in enumerate(shuffled,1):
        value=keyed_results[condition][case['case_id']];rid=f'R{number:03d}'
        row={k:case[k] for k in ['case_id','category','control_type','split','messages','source_url','source_block_ids','reference_checks','reference_evidence','composition_date_interval']}
        row.update(review_id=rid,response=value['response'],finish_reason=value['finish_reason'],token_count=value['token_count'],
                   composition_date_qualification=case.get('composition_date_qualification',''),reference_role_notes=case.get('reference_role_notes',''),**dict.fromkeys(SCORE_FIELDS,''))
        reviews.append(row)
        keys.append({'review_id':rid,'case_id':case['case_id'],'category':case['category'],'control_type':case['control_type'],
                     'condition':condition,'shard':shard_for[case['case_id']],'identity_sha256':manifest['identity_sha256'],
                     'response_sha256':digest(value['response'].encode()),'prompt_sha256':value['prompt_sha256']})
    target.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.data-revision-review-',dir=target.parent))
    try:
        (temp/'review_rows.jsonl').write_text(_lines(reviews));(temp/'review_key.jsonl').write_text(_lines(keys))
        for n in range(1,shards+1):(temp/f'review_shard_{n:02d}.jsonl').write_text(_lines([r for r in reviews if shard_for[r['case_id']]==n]))
        with (temp/'review_rows.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(reviews[0]));writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in reviews)
        metadata={'schema_version':2,'identity_sha256':manifest['identity_sha256'],'seed':seed,'shards':shards,'cases':48,'answers':144,
          'quality_scores_assigned':False,'review_only_evidence':True,'verification':verification,
          'assignment':'Deterministic SHA256 shuffle; neutral IDs; all three answers to each case remain in one shard.',
          'limitations':'Condition labels hidden, but prose may reveal condition. Coding-agent review is not independent historian validation. Fresh and inspected regression cases are reported separately. Same data/order/exposure; learning rate alone changes.',
          'source_sha256':{name:_file_hash(path) for name,path in source_paths.items()},'output_sha256':{p.name:_file_hash(p) for p in sorted(temp.iterdir())}}
        (temp/'review_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
        if target.exists():raise FileExistsError('Review directory appeared during export')
        temp.rename(target)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path);p.add_argument('--project',type=Path,default=PROJECT)
    p.add_argument('--output',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--seed',type=int,default=20261024);p.add_argument('--shards',type=int,default=3)
    a=p.parse_args();print(export_review(a.run_dir,project=a.project,output_dir=a.output,seed=a.seed,shards=a.shards,archive_path=a.archive))
