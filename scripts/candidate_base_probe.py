"""One bounded, unscored, sharded-base memory probe. No training or adapters."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
import time

CANDIDATE={'model_id':'Qwen/Qwen3.5-27B','revision':'fc05daec18b0a78c049392ed2e771dde82bdf654'}
GIB=1024**3


def sha(raw):return hashlib.sha256(raw).hexdigest()


def validate_protocol(protocol):
    device_map_for(protocol.get('placement_policy','auto'))
    if (protocol['purpose']!='candidate_memory_probe_only' or protocol['model']!=CANDIDATE
            or protocol['maximum_model_calls']!=1 or protocol['training_steps']!=0
            or protocol['adapters_loaded']!=[] or protocol['runtime_timeout_seconds']!=2700
            or protocol['max_input_tokens']!=5120 or protocol['max_output_tokens']!=16
            or protocol['per_gpu_weight_budget_bytes']!=13*GIB or protocol['gpu_ids']!=[0,1]
            or protocol['enable_thinking'] is not False or protocol['weight_file_bytes']!=55563022432
            or protocol['finite_logits_required'] is not True):
        raise ValueError('Candidate probe scope differs from frozen limits')


def device_map_for(policy,num_layers=64):
    if policy=='auto':return 'auto'
    if policy!='fixed_34_30' or num_layers!=64:raise ValueError('Unknown candidate placement policy or architecture')
    return {'model.embed_tokens':0,**{'model.layers.'+str(i):0 if i<34 else 1 for i in range(64)},
        'model.norm':1,'model.rotary_emb':1,'lm_head':1}


def placement(model):
    if not getattr(model,'is_loaded_in_4bit',False) or getattr(model,'peft_config',None):
        raise ValueError('Expected quantized base without adapters')
    devices=set();bad=[]
    for name,param in model.named_parameters():
        device=str(param.device);devices.add(device)
        if device not in ('cuda:0','cuda:1') or param.requires_grad or 'lora_' in name:
            bad.append(name)
    if bad or devices!={'cuda:0','cuda:1'}:
        raise ValueError('All frozen parameters must be on both selected GPUs; no CPU/disk offload: '+str(bad[:8]))
    for device in getattr(model,'hf_device_map',{}).values():
        if str(device) not in ('0','1','cuda:0','cuda:1'):
            raise ValueError('Unsupported offload placement')
    return sorted(devices)


def resource_gate(devices,cache_free,ram_available,weight_bytes):
    if len(devices)!=2 or any(d['name']!='Tesla T4' or d['free_memory_bytes']<14*GIB for d in devices):
        raise ValueError('Two free T4 devices with probe headroom required')
    if ram_available<12*GIB:raise ValueError('Insufficient host RAM for streamed loading')
    if cache_free<int(weight_bytes*1.15)+GIB:raise ValueError('Insufficient model-cache disk space')


def run(bundle,output):
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Candidate inference remains on Kaggle')
    bundle,output=Path(bundle),Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((bundle/'protocol.json').read_text());validate_protocol(protocol)
    for name,expected in protocol['files'].items():
        if sha((bundle/name).read_bytes())!=expected:raise ValueError('Frozen candidate file changed')
    import fcntl
    handle=(output/'.run.lock').open('a');fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();result={'schema_version':1,'purpose':protocol['purpose'],'model':CANDIDATE,
        'protocol_sha256':sha((bundle/'protocol.json').read_bytes()),'status':'initializing','phase':'validate_environment',
        'model_calls':0,'model_weights_loaded':False,'training_steps':0,'adapters_loaded':[],
        'quality_evaluated':False,'active_model_changed':False,'enable_thinking':False}
    def save():
        result['seconds']=time.monotonic()-start
        p=output/'result.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(output/'result.json')
    def expired(*_):raise TimeoutError('Candidate probe runtime exceeded45minutes')
    previous=signal.signal(signal.SIGALRM,expired);signal.alarm(protocol['runtime_timeout_seconds'])
    try:
        actual={name:importlib.metadata.version(name) for name in protocol['packages']}
        if actual!=protocol['packages']:raise ValueError('Package stack differs')
        result['packages']=actual;save()
        import torch
        from huggingface_hub import snapshot_download
        from huggingface_hub.constants import HF_HUB_CACHE
        from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,GenerationConfig,AutoConfig,set_seed
        if torch.cuda.device_count()!=2:raise ValueError('Expected exactly two selected GPUs')
        devices=[]
        for i in (0,1):
            free,total=torch.cuda.mem_get_info(i);devices.append({'index':i,'name':torch.cuda.get_device_name(i),'free_memory_bytes':free,'total_memory_bytes':total})
            torch.cuda.reset_peak_memory_stats(i)
        cache=Path(HF_HUB_CACHE)
        while not cache.exists():cache=cache.parent
        ram=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        cache_free=shutil.disk_usage(cache).free
        result['resources_before']={'cache_free_bytes':cache_free,'ram_available_bytes':ram}
        resource_gate(devices,cache_free,ram,protocol['weight_file_bytes'])
        result['devices_before']=devices;result['phase']='download_pinned_candidate';save()
        snapshot=snapshot_download(CANDIDATE['model_id'],revision=CANDIDATE['revision'],token=False,
            allow_patterns=['*.json','*.jinja','*.txt','*.safetensors'],max_workers=2)
        if Path(snapshot).name!=CANDIDATE['revision']:raise ValueError('Candidate revision differs')
        for name,expected in protocol['tokenizer_files'].items():
            if sha((Path(snapshot)/name).read_bytes())!=expected:raise ValueError('Candidate tokenizer/config changed')
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        if sha(tokenizer.chat_template.encode())!=protocol['chat_template_sha256']:raise ValueError('Candidate chat template differs')
        probe=json.loads((bundle/'memory_probe.json').read_text())
        ids=tokenizer.apply_chat_template(probe['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if ids!=probe['input_token_ids'] or not 4900<=len(ids)<=5120:raise ValueError('Probe input reconstruction differs')
        set_seed(42);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
        config=AutoConfig.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        policy=protocol.get('placement_policy','auto')
        selected_map=device_map_for(policy,config.get_text_config(decoder=True).num_hidden_layers)
        result['placement_policy']=policy
        result['phase']='load_sharded_candidate';save()
        model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
            dtype=torch.float16,device_map=selected_map,max_memory={0:13*GIB,1:13*GIB},attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
        model.requires_grad_(False);model.eval();result['model_weights_loaded']=True
        result['parameter_devices']=placement(model);result['device_map']={name:str(device) for name,device in model.hf_device_map.items()}
        result['memory_after_load']={str(i):{'allocated_bytes':torch.cuda.memory_allocated(i),'reserved_bytes':torch.cuda.memory_reserved(i)} for i in (0,1)}
        if policy=='fixed_34_30' and any(v['allocated_bytes']>protocol['per_gpu_weight_budget_bytes'] for v in result['memory_after_load'].values()):
            raise ValueError('Fixed placement exceeds the unchanged per-GPU weight-memory budget')
        result['preparation']='Fresh inference-only loader; no PEFT preparation and no changes to the archived single-GPU helper'
        input_device=model.get_input_embeddings().weight.device
        if str(input_device) not in ('cuda:0','cuda:1'):raise ValueError('Input embedding placement differs')
        generation=GenerationConfig.from_model_config(model.config.get_text_config(decoder=True))
        generation.do_sample=False;generation.num_beams=1;generation.max_new_tokens=16;generation.use_cache=True
        generation.eos_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>');generation.pad_token_id=tokenizer.pad_token_id
        result['generation']=generation.to_dict();result['phase']='single_unscored_memory_probe';save()
        tensor=torch.tensor([ids],device=input_device);result['model_calls']=1
        with torch.inference_mode():
            generated=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),generation_config=generation,return_dict_in_generate=True,output_logits=True)
            tokens=generated.sequences[0,len(ids):].tolist()
        if not generated.logits or len(generated.logits)!=len(tokens) or any(not torch.isfinite(step).all().item() for step in generated.logits):
            raise ValueError('Probe produced non-finite or missing raw logits')
        result['finite_logits']=True
        for i in (0,1):torch.cuda.synchronize(i)
        if not tokens or len(tokens)>16:raise ValueError('Invalid probe output bounds')
        result['probe']={'purpose':'memory/placement only, not a chatbot answer','input_token_ids':ids,'token_ids':tokens,
            'response':tokenizer.decode(tokens,skip_special_tokens=True),
            'finish_reason':'stop' if tokens[-1]==generation.eos_token_id else 'length'}
        result['peak_allocated_bytes']={str(i):torch.cuda.max_memory_allocated(i) for i in (0,1)}
        result['status']='memory_probe_passed';result['phase']='complete'
    except Exception as exc:
        result['status']='blocked';result['error_type']=type(exc).__name__;result['error']=str(exc)
        if 'torch' in locals():
            try:
                result['memory_after_failure']={str(i):{'allocated_bytes':torch.cuda.memory_allocated(i),'reserved_bytes':torch.cuda.memory_reserved(i),'peak_allocated_bytes':torch.cuda.max_memory_allocated(i)} for i in (0,1)}
            except Exception:pass
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous);save()
    print(json.dumps({k:result[k] for k in ('purpose','status','phase','model_calls','quality_evaluated')}),flush=True)
    return result
