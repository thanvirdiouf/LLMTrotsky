"""Verify exact downloaded hardware-probe source and structurally inspect inventory."""
import argparse
import ast
import json
from pathlib import Path
import re
from build_ghost_hardware_probe import ROOT,sha


def verify(version='ghost_hardware_v1',root=ROOT):
    if not re.fullmatch(r'ghost_hardware_v[1-9][0-9]*',version):raise ValueError('Invalid hardware probe version')
    stage=root/'training/kaggle_cli'/version;run=root/'evaluation/runs'/version
    lock=json.loads((root/'evaluation'/version/'bundle_lock.json').read_text())
    raw=(stage/'kaggle_ghost_hardware.ipynb').read_bytes()
    if sha(raw)!=lock['notebook_sha256'] or sha((stage/'kernel-metadata.json').read_bytes())!=lock['metadata_sha256']:
        raise ValueError('Frozen probe notebook or metadata changed')
    metadata=json.loads((stage/'kernel-metadata.json').read_text())
    if metadata['is_private'] is not True or metadata['enable_internet'] is not False or metadata['id_no']!=140792344:
        raise ValueError('Unexpected probe target or network policy')
    cell=next(c for c in json.loads(raw)['cells'] if c['id']=='probe');tree=ast.parse(''.join(cell['source']))
    source=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='probe_source' for t in n.targets))).encode()
    downloaded=(run/'download/ghost_hardware_bundle/ghost_hardware_probe.py').read_bytes()
    if source!=downloaded or sha(source)!=lock['probe_source_sha256']:raise ValueError('Downloaded probe source differs')
    result=json.loads((run/'download/ghost_hardware_results/result.json').read_text())
    if (result['schema_version'] not in (1,2) or result['purpose']!='hardware_inventory_only' or result['probe_source_sha256']!=sha(source)
            or result['model_weights_loaded'] is not False or type(result['model_calls']) is not int or result['model_calls']!=0
            or result['training_steps']!=0 or result['adapters_loaded']!=[]):raise ValueError('Unexpected probe result')
    devices=result['devices']
    if not isinstance(devices,list) or not devices or type(result['gpu_count']) is not int or result['gpu_count']!=len(devices):
        raise ValueError('Missing or inconsistent GPU inventory')
    for index,device in enumerate(devices):
        if device['index']!=index or device['name']!='Tesla T4':raise ValueError('Expected free Kaggle T4 inventory')
        for key in ('total_memory_bytes','free_memory_bytes','reported_total_bytes'):
            if type(device[key]) is not int or device[key]<0:raise ValueError('Invalid GPU memory count')
        if not 0<=device['free_memory_bytes']<=device['reported_total_bytes']<=device['total_memory_bytes']:
            raise ValueError('Inconsistent GPU memory inventory')
    for values in (result['ram_bytes'],result['working_disk_bytes']):
        if not values or any(type(v) is not int or v<0 for v in values.values()):raise ValueError('Invalid RAM/disk inventory')
    storage=result.get('storage_locations',{})
    if result['schema_version']==2:
        expected={'model_cache_filesystem','home_filesystem','temporary_filesystem'}
        if set(storage)!=expected or result.get('cache_location_source') not in ('huggingface_hub.constants','fallback_default_not_sdk_verified'):
            raise ValueError('Missing model-cache filesystem inventory')
        for value in storage.values():
            if (type(value.get('total')) is not int or type(value.get('free')) is not int
                    or not 0<=value['free']<=value['total'] or type(value.get('same_filesystem_as_working')) is not bool):
                raise ValueError('Invalid storage inventory')
    return {'status':'verified_inventory','model_calls':0,'model_weights_loaded':False,'gpu_count':len(devices),'devices':devices,
        'ram_bytes':result['ram_bytes'],'working_disk_bytes':result['working_disk_bytes'],
        **({'storage_locations':storage,'cache_location_source':result['cache_location_source']} if result['schema_version']==2 else {}),
        'larger_model_fit_established':False,'result_sha256':sha((run/'download/ghost_hardware_results/result.json').read_bytes())}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_hardware_v1');args=parser.parse_args()
    result=verify(args.version);out=ROOT/'evaluation/runs'/args.version/'verification.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
