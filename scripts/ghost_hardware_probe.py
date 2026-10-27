"""Read-only Kaggle hardware inventory: no model weights, training or inference."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil


def probe(output):
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Hardware probe runs only on Kaggle')
    import torch
    devices=[]
    for index in range(torch.cuda.device_count()):
        properties=torch.cuda.get_device_properties(index)
        free,total=torch.cuda.mem_get_info(index)
        devices.append({'index':index,'name':properties.name,'total_memory_bytes':properties.total_memory,
            'free_memory_bytes':free,'reported_total_bytes':total,'compute_capability':[properties.major,properties.minor]})
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemTotal','MemAvailable'):memory[key]=int(value.strip().split()[0])*1024
    disk=shutil.disk_usage('/kaggle/working')
    working_device=Path('/kaggle/working').stat().st_dev
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache=Path(HF_HUB_CACHE);cache_source='huggingface_hub.constants'
    except ImportError:
        cache=Path(os.path.expanduser('~'))/'.cache/huggingface/hub';cache_source='fallback_default_not_sdk_verified'
    while not cache.exists():cache=cache.parent
    storage={}
    for label,path in [('model_cache_filesystem',cache),('home_filesystem',Path(os.path.expanduser('~'))),('temporary_filesystem',Path('/tmp'))]:
        usage=shutil.disk_usage(path)
        storage[label]={'total':usage.total,'free':usage.free,'same_filesystem_as_working':path.stat().st_dev==working_device}

    result={'schema_version':2,'purpose':'hardware_inventory_only','probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'gpu_count':len(devices),'devices':devices,'ram_bytes':memory,'working_disk_bytes':{'total':disk.total,'free':disk.free},
        'storage_locations':storage,'cache_location_source':cache_source,
        'software':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda},
        'model_weights_loaded':False,'model_calls':0,'training_steps':0,'adapters_loaded':[]}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');os.replace(temporary,output)
    print(json.dumps({'purpose':result['purpose'],'gpu_count':len(devices),'model_calls':0}))
    return result
