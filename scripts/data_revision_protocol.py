"""Portable frozen guards for the data-only, 50-update intervention."""
import hashlib
import json
import math
import random

if __package__:
    from .build_training_pilot import validate_training_payload
else:
    from build_training_pilot import validate_training_payload

CONDITIONS = ['fresh_base', 'original_step_50', 'revised_step_50']
MODEL = {'model_id': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'}
GENERATION = {'do_sample': False, 'eos_token_id': 248046, 'max_new_tokens': 512,
              'num_beams': 1, 'pad_token_id': 248044, 'use_cache': True}
CATEGORIES = {'chronology', 'false_premise', 'invented_memory_citation', 'answer_detail'}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode())


def validate_settings(settings, original):
    if not isinstance(original, dict) or not original:
        raise ValueError('Missing original optimizer/model settings')
    expected = dict(original)
    expected['steps'] = 50
    if (not isinstance(settings, dict) or settings != expected
            or any(type(settings[k]) is not type(v) for k, v in expected.items())):
        raise ValueError('Only the fixed 50-update budget may differ from original optimizer/model settings')
    if (original.get('steps') != 500 or original.get('model_id') != MODEL['model_id']
            or original.get('revision') != MODEL['revision'] or original.get('seed') != 42
            or original.get('learning_rate') != 0.0001 or original.get('rank') != 8
            or original.get('lora_alpha') != 16 or original.get('lora_dropout') != 0.0
            or original.get('quantization') != 'nf4_double'
            or original.get('micro_batch_size') != 1
            or original.get('gradient_accumulation_steps') != 1
            or original.get('run_evaluation') is not True):
        raise ValueError('Original settings do not identify the verified control recipe')


def build_order(encoded, seed=42):
    if type(seed) is not int or seed != 42:
        raise ValueError('Training order requires the frozen seed 42')
    if not isinstance(encoded, (list, tuple)) or len(encoded) != 50:
        raise ValueError('Exactly 50 unique complete slice examples are required')
    ids = []
    for value in encoded:
        if (not isinstance(value, dict) or not isinstance(value.get('example_id'), str)
                or not value['example_id'].strip() or value.get('dropped') is not False
                or type(value.get('token_count')) is not int or value['token_count'] <= 0):
            raise ValueError('Invalid or incomplete encoded slice example')
        ids.append(value['example_id'])
    if len(set(ids)) != 50:
        raise ValueError('Exactly 50 unique complete slice examples are required')
    longest = max(range(50), key=lambda i: encoded[i]['token_count'])
    rest = [i for i in range(50) if i != longest]
    random.Random(seed).shuffle(rest)
    return [longest] + rest


def validate_payloads(train_bytes, eval_bytes, lock, protocol, pilot_manifest):
    identity = pilot_manifest.get('identity')
    if (not isinstance(identity, dict) or pilot_manifest.get('status') != 'complete'
            or pilot_manifest.get('completed_steps') != 500
            or pilot_manifest.get('identity_sha256') != fingerprint(identity)
            or not pilot_manifest.get('checks')
            or not all(value is True for value in pilot_manifest['checks'].values())):
        raise ValueError('Original pilot is incomplete or has a changed identity/checks')
    validate_settings(protocol['settings'], identity['settings'])
    if protocol['conditions'] != CONDITIONS or protocol['packages'] != identity['packages']:
        raise ValueError('Frozen conditions or package stack changed')
    if (protocol['pilot_identity'] != identity
            or fingerprint(protocol['pilot_identity']) != protocol['pilot_identity_sha256']
            or identity.get('model_id') != MODEL['model_id'] or identity.get('revision') != MODEL['revision']
            or identity.get('dtype') != 'torch.float16' or identity.get('gpu_ids_used') != [0]
            or identity.get('chat_template_sha256') != protocol['chat_template_sha256']):
        raise ValueError('Original pilot identity changed')
    if protocol['evaluation'].get('generation') != GENERATION:
        raise ValueError('Evaluation decoding differs from the frozen control settings')
    if digest(train_bytes) != protocol['training']['sha256'] or digest(eval_bytes) != protocol['evaluation']['sha256']:
        raise ValueError('Frozen training/evaluation checksum mismatch')
    if fingerprint(lock) != protocol['training_lock_fingerprint']:
        raise ValueError('Frozen training review lock differs')
    rows = validate_training_payload(train_bytes, lock)
    if len(rows) != 50 or lock['dataset']['records'] != 50:
        raise ValueError('Training must contain exactly 50 reviewed slice rows')
    eval_rows = [json.loads(s) for s in eval_bytes.splitlines()]
    if len(eval_rows) != 24:
        raise ValueError('Expected 24 unique evaluation cases')
    seen = set()
    for row in eval_rows:
        if (not isinstance(row, dict) or set(row) != {'case_id', 'category', 'split', 'messages'}
                or row['split'] != 'fresh_diagnostic_holdout_v2'
                or not isinstance(row['case_id'], str) or not row['case_id'].strip()
                or row['case_id'] in seen or row['category'] not in CATEGORIES):
            raise ValueError('Evaluation input allowlist/split/ID/category mismatch')
        seen.add(row['case_id'])
        messages = row['messages']
        if (not isinstance(messages, list) or len(messages) != 2
                or any(not isinstance(message, dict) or set(message) != {'role', 'content'}
                       or not isinstance(message['content'], str) or not message['content'].strip()
                       for message in messages)
                or [message['role'] for message in messages] != ['system', 'user']):
            raise ValueError('Evaluation must contain only nonempty system/user text')
        if digest(messages[0]['content'].encode()) != lock['persona']['system_prompt_sha256']:
            raise ValueError('Evaluation persona changed')
    return rows, eval_rows


def verify_metrics(metrics, order_ids):
    if (len(order_ids) != 50 or len(set(order_ids)) != 50 or len(metrics) != 50
            or [r.get('example_id') for r in metrics] != order_ids
            or [r.get('step') for r in metrics] != list(range(1, 51))
            or [r.get('verified_optimizer_step') for r in metrics] != list(range(1, 51))):
        raise ValueError('Must verify exactly 50 actual optimizer updates in frozen order')
    for row in metrics:
        if (row.get('finite_gradients') is not True or row.get('frozen_gradients_absent') is not True
                or type(row.get('input_tokens')) is not int or row['input_tokens'] < 1
                or type(row.get('supervised_tokens')) is not int
                or not 0 < row['supervised_tokens'] <= row['input_tokens']):
            raise ValueError('Training gradient/token evidence incomplete')
        for key in ['loss', 'gradient_norm_before_clip', 'grad_scaler_scale']:
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Training metrics contain invalid numerical evidence')
        if row['gradient_norm_before_clip'] <= 0 or row['grad_scaler_scale'] <= 0:
            raise ValueError('Adapter gradients or loss scale are not positive')
    return True
