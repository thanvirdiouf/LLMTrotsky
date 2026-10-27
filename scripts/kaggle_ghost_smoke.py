"""Memory-first present-day ghost smoke on the pinned Kaggle base; no training."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from grounded_chat import MODEL, fingerprint, load_collection
from evaluate_ghost_smoke import evaluate
from kaggle_grounding import atomic_json


def run(bundle, output):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Model inference stays on Kaggle; local loading is disabled')
    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output/'.run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((bundle/'protocol.json').read_text())
    for name, expected in protocol['files'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen bundle changed: '+name)
    if protocol['model'] != MODEL or protocol['training_steps'] != 0 or protocol['adapters_loaded'] != []:
        raise ValueError('Expected untouched base inference only')
    actual = {name: importlib.metadata.version(name) for name in protocol['packages']}
    if actual != protocol['packages']:
        raise ValueError('Package stack changed: '+str(actual))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, AutoConfig, set_seed
    from huggingface_hub import snapshot_download
    from peft import prepare_model_for_kbit_training
    from kbit_preparation import prepare_frozen_quantized_base
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'Tesla T4':
        raise RuntimeError('Expected Kaggle GPU0 Tesla T4')
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    set_seed(42)
    snapshot = snapshot_download(MODEL['model_id'], revision=MODEL['revision'],
        allow_patterns=['*.json', '*.jinja', '*.txt', '*.safetensors'], max_workers=2)
    if Path(snapshot).name != MODEL['revision']:
        raise ValueError('Base revision changed')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != protocol['chat_template_sha256']:
        raise ValueError('Chat template changed')
    collection = json.loads((bundle/'collection.json').read_text())
    profile = json.loads((bundle/'profile.json').read_text())
    spec = json.loads((bundle/'model_cases.json').read_text())
    thinking=protocol.get('enable_thinking',False)
    if type(thinking) is not bool or (thinking and (protocol['purpose']!='ghost_audit_calibration' or spec.get('audit_calibration',{}).get('policy')!='neutral_source' or spec['audit_calibration'].get('thinking') is not True)):
        raise ValueError('Unsupported thinking policy')
    evaluator=evaluate
    if protocol['purpose']=='ghost_direct_checked_comparison':
        from evaluate_ghost_comparison import evaluate as evaluator
    elif protocol['purpose']=='ghost_audit_calibration':
        from evaluate_ghost_audit import evaluate as evaluator
    elif protocol['purpose']=='ghost_factual_comparison':
        from evaluate_ghost_factual_comparison import evaluate as evaluator
    ledger = json.loads((bundle/'claims.json').read_text())
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
    generation = GenerationConfig.from_model_config(config.get_text_config(decoder=True))
    generation.temperature, generation.top_p, generation.top_k = 1., 1., 0
    for key, value in protocol['generation'].items():
        setattr(generation, key, value)
    if tokenizer.convert_tokens_to_ids('<|im_end|>') != generation.eos_token_id or tokenizer.pad_token_id != generation.pad_token_id:
        raise ValueError('Stop/pad identity changed')
    identity = {'protocol_sha256': hashlib.sha256((bundle/'protocol.json').read_bytes()).hexdigest(),
                'model': MODEL, 'packages': actual, 'generation': generation.to_dict(),
                'gpu': torch.cuda.get_device_name(0), 'gpu_ids_used': [0], 'seed': 42,
                'quantization': 'NF4 double quantization; float16 compute; sdpa',
                'adapters_loaded': [], 'optimizer_updates': 0, 'enable_thinking': thinking,
                'purpose': protocol['purpose']}
    if 'selection_constraints' in protocol:identity['selection_constraints']=protocol['selection_constraints']
    manifest_path = output/'run_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['identity'] != identity:
        raise ValueError('Resume identity changed')
    atomic_json(manifest_path, {'status': 'loading', 'identity': identity})
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
        trust_remote_code=False, dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
    model = prepare_frozen_quantized_base(model, dtype=torch.float16, device=torch.device('cuda', 0),
                                         prepare_kbit=prepare_model_for_kbit_training, training=False)
    if not getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'peft_config', None) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Base must be frozen with no adapter')
    model.eval()
    constraint_cache={}
    def generate(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=thinking, return_dict=False)
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError('Invalid token list')
        if len(ids)>protocol['max_input_tokens']:
            return {'response':'','finish_reason':'input_budget','input_token_ids':ids,'token_ids':[],
                    'token_count':0,'seconds':0,'model_call_made':False}
        torch.cuda.empty_cache()
        tensor = torch.tensor([ids], device='cuda:0')
        constraint=None;extra={}
        if 'selection_constraints' in protocol:
            from selection_constraints import selection_ids,SelectionTrie
            available=selection_ids(messages)
            if available is not None:
                key=tuple(available)
                if key not in constraint_cache:
                    constraint_cache[key]=SelectionTrie(tokenizer,available,generation.eos_token_id,protocol['selection_constraints'])
                constraint=constraint_cache[key]
                if constraint.max_tokens>generation.max_new_tokens:raise ValueError('Selection grammar exceeds generation cap')
                extra['prefix_allowed_tokens_fn']=constraint.callback(len(ids))
        start = time.monotonic()
        with torch.inference_mode():
            tokens = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), generation_config=generation,**extra)[0, len(ids):].tolist()
        torch.cuda.synchronize()
        if constraint is not None and not constraint.accepts(tokens):raise ValueError('Selection output violates its finite grammar')
        result={'response': tokenizer.decode(tokens, skip_special_tokens=True),
                'finish_reason': 'stop' if tokens[-1] == generation.eos_token_id else 'length',
                'input_token_ids': ids, 'token_ids': tokens, 'token_count': len(tokens),
                'seconds': time.monotonic()-start,'model_call_made':True}
        if constraint is not None:result['selection_constraint']=constraint.metadata
        return result
    if not (output/'memory_probe.json').exists():
        probe=json.loads((bundle/'memory_probe.json').read_text())
        saved_limit=generation.max_new_tokens
        generation.max_new_tokens=16
        try:
            result=generate(probe['messages'])
        except Exception as exc:
            atomic_json(manifest_path, {'status':'memory_probe_failed','identity':identity,'error':str(exc)})
            raise
        finally:
            generation.max_new_tokens=saved_limit
        atomic_json(output/'memory_probe.json',{'purpose':'memory_only; not a scored answer','result':result,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(0)})
        if result.get('model_call_made') is not True:
            raise ValueError('Memory probe exceeds configured input budget')
        if result['input_token_ids']!=probe['input_token_ids']:
            raise ValueError('Memory probe tokenizer reconstruction differs')
        print('Memory probe passed',len(result['input_token_ids']),'input tokens',flush=True)
    try:
        rows, calls = evaluator(profile, collection, ledger, spec, generate, output, identity)
    except Exception as exc:
        atomic_json(manifest_path, {'status': 'incomplete', 'identity': identity, 'error': str(exc)})
        raise
    expected = len(spec['cases'])+len(spec['conversation'])
    if len(rows) != expected or len(rows) != protocol['answers'] or len(calls) > protocol['maximum_model_calls']:
        raise ValueError('Answer count differs')
    atomic_json(manifest_path, {'status': 'complete', 'identity': identity, 'answers': len(rows),
        'answers_sha256': hashlib.sha256((output/'answers.jsonl').read_bytes()).hexdigest(),
        'model_calls_sha256': hashlib.sha256((output/'model_calls.jsonl').read_bytes()).hexdigest(),
        'model_calls': len(calls),
        'generation_calls_made': sum(c['output'].get('model_call_made') is True for c in calls),
        'blocked_input_requests': sum(c['output'].get('model_call_made') is False for c in calls),
        'quality_scores_assigned': False, 'all_base_parameters_frozen': True,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(0)})
