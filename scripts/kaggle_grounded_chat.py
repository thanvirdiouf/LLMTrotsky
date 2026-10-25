"""Base-only end-to-end retrieval and sustained conversation inference on Kaggle."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from grounded_chat import MODEL, fingerprint, load_collection
from evaluate_grounded_chat import evaluate
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
    collection = load_collection(bundle/'collection.json')
    spec = json.loads((bundle/'model_cases.json').read_text())
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
                'adapters_loaded': [], 'optimizer_updates': 0, 'enable_thinking': False}
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
    if getattr(model, 'peft_config', None) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Base must be frozen with no adapter')
    model.eval()
    def generate(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_dict=False)
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids) or len(ids) > protocol['max_input_tokens']:
            raise ValueError('Input exceeds budget or invalid token list; no truncation')
        tensor = torch.tensor([ids], device='cuda:0')
        start = time.monotonic()
        with torch.inference_mode():
            tokens = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), generation_config=generation)[0, len(ids):].tolist()
        torch.cuda.synchronize()
        return {'response': tokenizer.decode(tokens, skip_special_tokens=True),
                'finish_reason': 'stop' if tokens[-1] == generation.eos_token_id else 'length',
                'input_token_ids': ids, 'token_ids': tokens, 'token_count': len(tokens),
                'seconds': time.monotonic()-start}
    try:
        rows = evaluate(collection, spec, generate, output/'responses.jsonl', identity)
    except Exception as exc:
        atomic_json(manifest_path, {'status': 'incomplete', 'identity': identity, 'error': str(exc)})
        raise
    expected = 2*len(spec['singles']) + sum(len(c['turns']) for c in spec['conversations'])
    if len(rows) != expected:
        raise ValueError('Answer count differs')
    atomic_json(manifest_path, {'status': 'complete', 'identity': identity, 'answers': len(rows),
        'responses_sha256': hashlib.sha256((output/'responses.jsonl').read_bytes()).hexdigest(),
        'quality_scores_assigned': False, 'all_base_parameters_frozen': True,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(0)})
