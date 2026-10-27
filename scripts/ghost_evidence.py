"""Model-free inspection of complete bounded source packets, not an answer judge."""
from copy import deepcopy
import json
import re
from urllib.parse import urlsplit,quote
try:
    from .grounded_chat_v2 import EvidenceError,initial_state
except ImportError:
    from grounded_chat_v2 import EvidenceError,initial_state

NOTICE=('These are source passages, not a generated answer or independent verification. '
        'Retrieval and source selection may miss evidence or include irrelevant material. '
        'Retain the original speaker, date and qualifications when interpreting a passage.')


def evidence_view(bot,question='',packet_ids=None,*,state=None):
    """Return original packet contents without altering conversation or trusted memory."""
    if not isinstance(question,str):raise EvidenceError('Question must be text')
    before=deepcopy(state if state is not None else initial_state(bot.collection,bot.ledger))
    if packet_ids is not None:
        packets={p['id']:p for p in bot.collection['packets']}
        if (not isinstance(packet_ids,list) or not 1<=len(packet_ids)<=2
                or any(not isinstance(pid,str) for pid in packet_ids)
                or len(set(packet_ids))!=len(packet_ids) or set(packet_ids)-packets.keys()):
            raise EvidenceError('Select one or two distinct known source IDs')
        selected=[deepcopy(packets[pid]) for pid in packet_ids]
        retrieval={'method':'explicit_source_inspection','packet_ids':list(packet_ids),'budget_omitted':[]}
    else:
        if not question.strip():raise EvidenceError('A question or explicit source IDs are required')
        trace=bot.retriever.retrieve(question,before)
        selected=[deepcopy(m['packet']) for m in trace['matches']]
        retrieval={k:deepcopy(trace[k]) for k in ('method','budget_omitted','explicit_packet_ids','retained_packet_ids','topic_reset') if k in trace}
        retrieval['packet_ids']=[p['id'] for p in selected]
    return {'schema_version':1,'mode':'source_evidence_view','reference_date':bot.profile['reference_date'],
        'question':question,'notice':NOTICE,'retrieval':retrieval,'source_packets':selected,
        'generated_answer':None,'model_calls':0,'semantic_verification':False,
        'state_before':before,'state_after':deepcopy(before)}


def literal(text):
    """Fence source text without allowing embedded fences to close the block."""
    runs=re.findall(r'~+',text);fence='~'*max(3,1+max((len(r) for r in runs),default=0))
    return fence+'text\n'+text+'\n'+fence


def render_evidence(view):
    parts=['Source evidence',view['notice'],f"Frozen reference date: {view['reference_date']}"]
    if not view['source_packets']:parts.append('No source packet was retrieved from this bounded collection.')
    for packet in view['source_packets']:
        parts.append(f"Source ID: {packet['id']}")
        metadata={k:packet[k] for k in ('title','author','domain','composition_date_interval','composition_date_evidence',
            'observed_date','context_boundary') if k in packet}
        parts.append(literal(json.dumps(metadata,ensure_ascii=False,indent=2)))
        url=packet.get('url','');parsed=urlsplit(url)
        if parsed.scheme in ('http','https') and parsed.netloc:
            parts.append('[Original source](<'+quote(url,safe='/:?&=%#')+'>)')
        for block in packet['blocks']:
            parts.append(literal(f"Block {block['block_id']} | Speaker: {block['speaker']}\n{block['text']}"))
    if view['retrieval'].get('budget_omitted'):
        parts.append('Source packets omitted by the retrieval budget: '+', '.join(view['retrieval']['budget_omitted']))
    return '\n\n'.join(parts)+'\n'
