"""Check exact frozen candidate probe artifacts; memory success is not quality."""
import argparse
import ast
import base64
import json
import lzma
from pathlib import Path
import re
from build_ghost_candidate_probe import ROOT,sha
from candidate_base_probe import CANDIDATE,validate_protocol,resource_gate


def payloads(version,root=ROOT):
    if not re.fullmatch(r'ghost_candidate_probe_v[1-9][0-9]*',version):raise ValueError('Invalid candidate probe version')
    stage=root/'training/kaggle_cli'/version;lock=json.loads((root/'evaluation'/version/'bundle_lock.json').read_text())
    raw=(stage/'kaggle_ghost_candidate_probe.ipynb').read_bytes()
    if sha(raw)!=lock['notebook_sha256'] or sha((stage/'kernel-metadata.json').read_bytes())!=lock['metadata_sha256']:
        raise ValueError('Frozen notebook or metadata differs')
    metadata=json.loads((stage/'kernel-metadata.json').read_text())
    if metadata['is_private'] is not True or metadata['id_no']!=140792344:raise ValueError('Wrong candidate target')
    tree=ast.parse(''.join(next(c for c in json.loads(raw)['cells'] if c['id']=='run')['source']))
    packed=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='packed' for t in n.targets)))
    values={name:lzma.decompress(base64.b85decode(data)) for name,data in packed.items()}
    if {name:sha(raw) for name,raw in values.items()}!=lock['payload_sha256']:raise ValueError('Frozen payload differs')
    return values


def verify(version='ghost_candidate_probe_v1',root=ROOT,tokenizer=None):
    frozen=payloads(version,root);protocol=json.loads(frozen['protocol.json']);validate_protocol(protocol)
    run=root/'evaluation/runs'/version;download=run/'download'
    for name,raw in frozen.items():
        if (download/'ghost_candidate_bundle'/name).read_bytes()!=raw:raise ValueError('Downloaded candidate payload differs: '+name)
    for name,digest in protocol['preserved_files'].items():
        if sha((root/name).read_bytes())!=digest:raise ValueError('Protected archive changed')
    result_path=download/'ghost_candidate_results/result.json';result=json.loads(result_path.read_text())
    if (result['model']!=CANDIDATE or result['protocol_sha256']!=sha(frozen['protocol.json'])
            or result['purpose']!=protocol['purpose'] or result['training_steps']!=0 or result['adapters_loaded']!=[]
            or result['quality_evaluated'] is not False or result['active_model_changed'] is not False
            or result['enable_thinking'] is not False or type(result['model_calls']) is not int or not 0<=result['model_calls']<=1):
        raise ValueError('Candidate scope/result differs')
    if result.get('packages') is not None and result['packages']!=protocol['packages']:raise ValueError('Candidate packages differ')
    report={'status':'verified_probe_result','probe_status':result['status'],'model':CANDIDATE,'model_calls':result['model_calls'],
        'quality_evaluated':False,'active_model_changed':False,'result_sha256':sha(result_path.read_bytes())}
    if result['status']=='blocked':
        if not result.get('error_type') or not result.get('error'):raise ValueError('Missing blocked-probe reason')
        report.update(error_type=result['error_type'],error=result['error'],phase=result['phase'],memory_success=False)
        return report
    if result['status']!='memory_probe_passed' or result['model_calls']!=1 or result['model_weights_loaded'] is not True or result.get('finite_logits') is not True:
        raise ValueError('Candidate probe incomplete')
    resources=result['resources_before'];resource_gate(result['devices_before'],resources['cache_free_bytes'],resources['ram_available_bytes'],protocol['weight_file_bytes'])
    if result['parameter_devices']!=['cuda:0','cuda:1'] or any(v not in ('0','1','cuda:0','cuda:1') for v in result['device_map'].values()):
        raise ValueError('Candidate placement differs')
    probe=result['probe'];expected=json.loads(frozen['memory_probe.json'])
    if probe['input_token_ids']!=expected['input_token_ids'] or not 4900<=len(probe['input_token_ids'])<=5120:
        raise ValueError('Candidate probe inputs differ')
    ids=probe['token_ids'];generation=result['generation']
    if (not isinstance(ids,list) or not 1<=len(ids)<=16 or any(type(i) is not int or i<0 for i in ids)
            or generation['do_sample'] is not False or generation['num_beams']!=1 or generation['max_new_tokens']!=16 or generation['use_cache'] is not True):
        raise ValueError('Invalid candidate output/generation')
    finish='stop' if ids[-1]==generation['eos_token_id'] else 'length'
    if probe['finish_reason']!=finish or (finish=='length' and len(ids)!=16):raise ValueError('Candidate stop differs')
    peaks=result['peak_allocated_bytes']
    if set(peaks)!={'0','1'} or any(type(v) is not int or v<=0 for v in peaks.values()):raise ValueError('Missing GPU peaks')
    if any(peaks[str(d['index'])]>d['total_memory_bytes'] for d in result['devices_before']):raise ValueError('Impossible GPU peak')
    if tokenizer is not None:
        if sha(tokenizer.chat_template.encode())!=protocol['chat_template_sha256']:raise ValueError('Candidate tokenizer differs')
        if tokenizer.apply_chat_template(expected['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)!=probe['input_token_ids']:
            raise ValueError('Candidate tokenizer input reconstruction differs')
        if tokenizer.decode(ids,skip_special_tokens=True)!=probe['response']:raise ValueError('Candidate tokenizer output differs')
    report.update(memory_success=True,finite_logits=True,probe_input_tokens=len(probe['input_token_ids']),probe_output_tokens=len(ids),
        peak_gib={name:value/1024**3 for name,value in peaks.items()},tokenizer_reconstructed_locally=tokenizer is not None)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_candidate_probe_v1');parser.add_argument('--tokenizer',type=Path)
    args=parser.parse_args();tokenizer=None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    report=verify(args.version,tokenizer=tokenizer);output=ROOT/'evaluation/runs'/args.version/'verification.json';output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
