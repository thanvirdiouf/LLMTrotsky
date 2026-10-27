"""Candidate-only SDPA policy: explicit KV replication, efficient kernel only."""
from types import SimpleNamespace

NAME='candidate_efficient_sdpa'
POLICY='repeat_kv_efficient_sdpa'
ATOL=0.002
RTOL=0.01
CASES=('prefill_causal','decode','masked_prefill')


class NoNativeGQA:
    num_key_value_groups=1
    def __init__(self,module):self._module=module
    def __getattr__(self,name):return getattr(self._module,name)


def make_backend(sdpa_forward,repeat_kv,context,efficient_backend,stats):
    stats.update(calls=0,expanded_kv_calls=0,backend='EFFICIENT_ATTENTION',native_gqa=False,mask_function='sdpa_mask')
    def backend(module,query,key,value,attention_mask,**kwargs):
        if kwargs.get('cache') is not None:raise ValueError('Paged-cache attention is outside this candidate policy')
        if any(len(t.shape)!=4 for t in (query,key,value)):raise ValueError('Expected batched head-major attention tensors')
        groups=getattr(module,'num_key_value_groups',query.shape[1]//key.shape[1])
        if groups<1 or query.shape[1]!=key.shape[1]*groups or key.shape[1]!=value.shape[1]:
            raise ValueError('Invalid grouped attention dimensions')
        if groups>1:
            key=repeat_kv(key,groups);value=repeat_kv(value,groups)
        with context(efficient_backend):
            result=sdpa_forward(NoNativeGQA(module),query,key,value,attention_mask,**kwargs)
        stats['calls']+=1
        if groups>1:stats['expanded_kv_calls']+=1
        return result
    return backend


def register_with(attention_registry,mask_registry,mask_function,backend):
    attention_registry.register(NAME,backend)
    mask_registry.register(NAME,mask_function)
    return NAME


def register_backend(stats):
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from transformers import AttentionInterface,AttentionMaskInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward,repeat_kv
    from transformers.masking_utils import sdpa_mask
    backend=make_backend(sdpa_attention_forward,repeat_kv,lambda b:sdpa_kernel(backends=[b]),SDPBackend.EFFICIENT_ATTENTION,stats)
    register_with(AttentionInterface,AttentionMaskInterface,sdpa_mask,backend)
    return NAME,backend


def cuda_equivalence_checks(backend,torch):
    """Small synthetic kernel checks before any candidate weights are fetched."""
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    records=[]
    module=SimpleNamespace(num_key_value_groups=6,is_causal=True,layer_idx=0)
    with torch.inference_mode():
        for index in (0,1):
            device='cuda:'+str(index);generator=torch.Generator(device=device).manual_seed(42)
            for case in CASES:
                q_length,k_length=(1,65) if case=='decode' else (64,64)
                q=torch.randn((1,24,q_length,256),device=device,dtype=torch.float16,generator=generator)
                k=torch.randn((1,4,k_length,256),device=device,dtype=torch.float16,generator=generator)
                v=torch.randn((1,4,k_length,256),device=device,dtype=torch.float16,generator=generator)
                mask=None
                if case=='masked_prefill':
                    mask=torch.ones((q_length,k_length),device=device,dtype=torch.bool).tril()
                    mask[:,-1]=False;mask=mask[None,None,:,:]
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    expected=sdpa_attention_forward(module,q,k,v,mask,dropout=0.0,scaling=256**-0.5)[0]
                actual=backend(module,q,k,v,mask,dropout=0.0,scaling=256**-0.5)[0]
                torch.testing.assert_close(actual,expected,atol=ATOL,rtol=RTOL)
                if not torch.isfinite(actual).all().item():raise ValueError('Non-finite synthetic attention output')
                records.append({'device':index,'case':case,'passed':True,'max_abs_error':float((actual-expected).abs().max().item()),'atol':ATOL,'rtol':RTOL})
    return records
