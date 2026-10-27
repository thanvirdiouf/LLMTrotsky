"""Separate pinned27B inference diagnostic; fixed archived9B audit messages."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
import time
import traceback
from candidate_base_probe import CANDIDATE,GIB,sha,placement,resource_gate,device_map_for


def validate_protocol(p):
    exact={'purpose':'candidate_neutral_audit','model':CANDIDATE,'evaluation_calls':10,
        'memory_probe_calls':1,'max_input_tokens':5120,'max_output_tokens':1024,
        'memory_probe_output_tokens':16,'enable_thinking':False,'training_steps':0,
        'adapters_loaded':[],'active_model_changed':False,'gpu_ids':[0,1],
        'placement_policy':'fixed_34_30','attention_policy':'repeat_kv_efficient_sdpa',
        'per_gpu_weight_budget_bytes':13*GIB,'weight_file_bytes':55563022432,
        'runtime_timeout_seconds':2700,'installation_timeout_seconds':600}
    if any(p.get(k)!=v or type(p.get(k)) is not type(v) for k,v in exact.items()):
        raise ValueError('Candidate audit scope differs')
    g=p['generation']
    if g['do_sample'] is not False or g['num_beams']!=1 or g['max_new_tokens']!=1024 or g['use_cache'] is not True:
        raise ValueError('Candidate generation differs')


def input_ids(tokenizer,messages,limit=5120):
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    if not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 for i in ids) or len(ids)>limit:
        raise ValueError('Invalid or over-budget candidate input')
    return ids


def fresh_output(output):
    if any((output/name).exists() for name in ('result.json','judgments.jsonl','memory_probe.json')):
        raise ValueError('Existing candidate audit artifacts; do not rerun or resume automatically')


def run(bundle,output):
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Candidate inference stays on Kaggle')
    bundle,output=Path(bundle),Path(output);output.mkdir(parents=True,exist_ok=True)
    p=json.loads((bundle/'protocol.json').read_text());validate_protocol(p);fresh_output(output)
    for name,digest in p['files'].items():
        if sha((bundle/name).read_bytes())!=digest:raise ValueError('Frozen audit payload changed: '+name)
    import fcntl
    lock=(output/'.run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();result={'purpose':p['purpose'],'model':CANDIDATE,'protocol_sha256':sha((bundle/'protocol.json').read_bytes()),
        'status':'initializing','phase':'environment','evaluation_calls':0,'memory_probe_calls':0,
        'training_steps':0,'adapters_loaded':[],'active_model_changed':False,'enable_thinking':False,'quality_validated':False}
    def save():
        result['seconds']=time.monotonic()-started
        temp=output/'result.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(output/'result.json')
    def expired(*_):raise TimeoutError('Candidate audit exceeded45minutes; no automatic retry')
    previous=signal.signal(signal.SIGALRM,expired);signal.alarm(p['runtime_timeout_seconds'])
    try:
        result['packages']={name:importlib.metadata.version(name) for name in p['packages']}
        if result['packages']!=p['packages']:raise ValueError('Package stack differs')
        import torch
        from huggingface_hub import snapshot_download
        from huggingface_hub.constants import HF_HUB_CACHE
        from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM,BitsAndBytesConfig,GenerationConfig,set_seed
        from candidate_efficient_attention import register_backend,cuda_equivalence_checks
        if torch.cuda.device_count()!=2:raise ValueError('Expected exactly two GPUs')
        devices=[]
        for i in (0,1):
            free,total=torch.cuda.mem_get_info(i)
            devices.append({'index':i,'name':torch.cuda.get_device_name(i),'free_memory_bytes':free,'total_memory_bytes':total})
            torch.cuda.reset_peak_memory_stats(i)
        cache=Path(HF_HUB_CACHE)
        while not cache.exists():cache=cache.parent
        ram=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        result['resources_before']={'cache_free_bytes':shutil.disk_usage(cache).free,'ram_available_bytes':ram}
        resource_gate(devices,result['resources_before']['cache_free_bytes'],ram,p['weight_file_bytes']);result['devices_before']=devices
        result['phase']='attention_checks';save()
        stats={};attention,backend=register_backend(stats)
        result['attention_equivalence_checks']=cuda_equivalence_checks(backend,torch)
        stats['calls']=0;stats['expanded_kv_calls']=0;result['attention_runtime_stats']=stats
        for i in (0,1):
            with torch.cuda.device(i):torch.cuda.empty_cache()
        result['phase']='download_and_load';save()
        snapshot=snapshot_download(CANDIDATE['model_id'],revision=CANDIDATE['revision'],token=False,
            allow_patterns=['*.json','*.jinja','*.txt','*.safetensors'],max_workers=2)
        if Path(snapshot).name!=CANDIDATE['revision']:raise ValueError('Candidate revision differs')
        for name,digest in p['tokenizer_files'].items():
            if sha((Path(snapshot)/name).read_bytes())!=digest:raise ValueError('Candidate asset changed')
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        if sha(tokenizer.chat_template.encode())!=p['chat_template_sha256']:raise ValueError('Template differs')
        jobs=json.loads((bundle/'audit_inputs.json').read_text());probe=json.loads((bundle/'memory_probe.json').read_text())
        if len(jobs)!=10 or len({j['case_id'] for j in jobs})!=10:raise ValueError('Candidate coverage differs')
        for job in [probe,*jobs]:
            if input_ids(tokenizer,job['messages'])!=job['input_token_ids']:raise ValueError('Frozen input reconstruction differs')
        set_seed(42);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
        config=AutoConfig.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
            dtype=torch.float16,device_map=device_map_for(p['placement_policy'],config.get_text_config(decoder=True).num_hidden_layers),
            max_memory={0:13*GIB,1:13*GIB},attn_implementation=attention,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
        model.requires_grad_(False);model.eval();result['parameter_devices']=placement(model)
        result['device_map']={name:str(device) for name,device in model.hf_device_map.items()}
        result['memory_after_load']={str(i):{'allocated_bytes':torch.cuda.memory_allocated(i),'reserved_bytes':torch.cuda.memory_reserved(i)} for i in (0,1)}
        if any(v['allocated_bytes']>p['per_gpu_weight_budget_bytes'] for v in result['memory_after_load'].values()):raise ValueError('Weight budget exceeded')
        input_device=model.get_input_embeddings().weight.device
        generation=GenerationConfig.from_dict(p['generation'])
        if generation.eos_token_id!=tokenizer.convert_tokens_to_ids('<|im_end|>') or generation.pad_token_id!=tokenizer.pad_token_id:raise ValueError('Stop or pad differs')
        result['generation']=generation.to_dict()
        class FiniteScores:
            def __init__(self):self.steps=0
            def __call__(self,input_ids,scores):
                if not torch.isfinite(scores).all().item():raise ValueError('Non-finite generation logits')
                self.steps+=1
                return scores
        def generate(job,cap):
            ids=input_ids(tokenizer,job['messages']);generation.max_new_tokens=cap
            for i in (0,1):
                with torch.cuda.device(i):torch.cuda.empty_cache()
            tensor=torch.tensor([ids],device=input_device);finite=FiniteScores();before=stats['calls'];before_groups=stats['expanded_kv_calls'];start=time.monotonic()
            with torch.inference_mode():
                tokens=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),generation_config=generation,logits_processor=[finite])[0,len(ids):].tolist()
            for i in (0,1):torch.cuda.synchronize(i)
            if not tokens or len(tokens)>cap or finite.steps!=len(tokens) or stats['calls']-before!=16*len(tokens) or stats['expanded_kv_calls']-before_groups!=16*len(tokens):raise ValueError('Generation or efficient attention coverage differs')
            return {'input_token_ids':ids,'token_ids':tokens,'token_count':len(tokens),'response':tokenizer.decode(tokens,skip_special_tokens=True),
                'finish_reason':'stop' if tokens[-1]==generation.eos_token_id else 'length','seconds':time.monotonic()-start,
                'finite_logits':True,'finite_logit_steps':finite.steps,'efficient_attention_calls':stats['calls']-before,'expanded_kv_calls':stats['expanded_kv_calls']-before_groups}
        result['phase']='memory_probe';result['memory_probe_calls']=1;save()
        probe_out=generate(probe,16);(output/'memory_probe.json').write_text(json.dumps(probe_out)+'\n')
        result['phase']='fixed_neutral_audit';save()
        with (output/'judgments.jsonl').open('x') as journal:
            for index,job in enumerate(jobs,1):
                result['evaluation_calls']=index;save()
                out=generate(job,1024)
                row={'case_id':job['case_id'],'model':CANDIDATE,'input_sha256':sha(json.dumps(job['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()),'output':out}
                journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
                print(job['case_id'],out['finish_reason'],len(out['token_ids']),flush=True)
        result['judgments_sha256']=sha((output/'judgments.jsonl').read_bytes());result['memory_probe_sha256']=sha((output/'memory_probe.json').read_bytes())
        result['status']='complete';result['phase']='complete'
    except Exception as exc:
        result['status']='blocked';result['error_type']=type(exc).__name__;result['error']=str(exc)
        result['error_frames']=[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno} for f in traceback.extract_tb(exc.__traceback__)[-8:]]
    finally:
        if 'torch' in locals():
            try:result['peak_allocated_bytes']={str(i):torch.cuda.max_memory_allocated(i) for i in (0,1)}
            except Exception:pass
        signal.alarm(0);signal.signal(signal.SIGALRM,previous);save()
    print(json.dumps({k:result[k] for k in ('status','phase','memory_probe_calls','evaluation_calls','quality_validated')}),flush=True)
    return result
