"""Finite, source-membership-constrained selection JSON; no semantic guarantees."""
import hashlib
from itertools import combinations,product
import json

POLICY={'policy':'finite_selection_v1','maximum_claims':4,'maximum_available_claims':12,
        'ordering':'priority_first_then_source_order'}


def selection_ids(messages):
    if (len(messages)!=2 or messages[0].get('role')!='system' or messages[1].get('role')!='user'
            or 'Select relevant supplied claim IDs.' not in messages[0]['content']):return None
    data=json.loads(messages[1]['content'])
    ids=[p['id'] for p in data['evidence_propositions']]
    if any(not isinstance(cid,str) or not cid for cid in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Selection constraint requires unique supplied claim IDs')
    return ids


def candidate_texts(claim_ids,policy=POLICY):
    if policy!=POLICY:raise ValueError('Unknown selection constraint policy')
    if any(not isinstance(cid,str) or not cid for cid in claim_ids):raise ValueError('Invalid supplied claim ID')
    if len(claim_ids)>policy['maximum_available_claims'] or len(set(claim_ids))!=len(claim_ids):
        raise ValueError('Unsupported available-claim count')
    # Every subset of up to four claims and every possible first priority is
    # available. Remaining IDs follow source order to keep this finite/bounded.
    orders=[[]]
    for first in range(len(claim_ids)):
        rest=[i for i in range(len(claim_ids)) if i!=first]
        for size in range(min(policy['maximum_claims']-1,len(rest))+1):
            for tail in combinations(rest,size):orders.append([claim_ids[i] for i in (first,*tail)])
    for mode,detail in product(('facts','view','mixed'),('brief','normal','expanded')):
        for ids in orders:
            yield json.dumps({'answer_mode':mode,'detail':detail,'claim_ids':ids},separators=(',',':'),ensure_ascii=False)


class SelectionTrie:
    def __init__(self,tokenizer,claim_ids,eos_token_id,policy=POLICY):
        if type(eos_token_id)is not int:raise ValueError('Expected a single EOS token')
        self.tree={};self.max_tokens=0;digest=hashlib.sha256();count=0
        batch=[]
        def insert(values):
            nonlocal count
            encoded=tokenizer(values,add_special_tokens=False)['input_ids']
            if len(encoded)!=len(values):raise ValueError('Tokenizer dropped constrained candidates')
            for text,ids in zip(values,encoded):
                if not ids or any(type(i)is not int or i==eos_token_id for i in ids):raise ValueError('Invalid constrained token path')
                node=self.tree
                for token in ids+[eos_token_id]:node=node.setdefault(token,{})
                self.max_tokens=max(self.max_tokens,len(ids)+1)
                digest.update(text.encode()+b'\n');count+=1
        for text in candidate_texts(claim_ids,policy):
            batch.append(text)
            if len(batch)==256:insert(batch);batch=[]
        if batch:insert(batch)
        self.metadata={**policy,'available_claim_ids':list(claim_ids),'candidate_count':count,
            'candidate_texts_sha256':digest.hexdigest(),'maximum_output_tokens':self.max_tokens}

    def allowed(self,generated):
        node=self.tree
        for token in generated:
            if token not in node:raise ValueError('Generated selection left the allowed token tree')
            node=node[token]
        if not node:raise ValueError('Selection already ended')
        return sorted(node)

    def accepts(self,tokens):
        node=self.tree
        for token in tokens:
            if token not in node:return False
            node=node[token]
        return not node

    def callback(self,prompt_tokens):
        def allowed_tokens(batch_id,input_ids):
            if batch_id!=0:raise ValueError('Constrained selection expects one sequence')
            generated=input_ids[prompt_tokens:]
            return self.allowed(generated.tolist() if hasattr(generated,'tolist') else list(generated))
        return allowed_tokens
