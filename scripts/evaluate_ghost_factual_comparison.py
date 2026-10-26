"""Five independent factual tasks in canonical and source-reading pipelines."""
try:
    from .ghost_chat import GhostChat
    from .source_reading import SourceReadingChat
    from .evaluate_ghost_smoke import evaluate as journal_evaluate
except ImportError:
    from ghost_chat import GhostChat
    from source_reading import SourceReadingChat
    from evaluate_ghost_smoke import evaluate as journal_evaluate


def evaluate(profile,collection,ledger,spec,generate,output,identity):
    if spec.get('conversation') or spec.get('factual_comparison',{}).get('arms')!=['canonical','reading']:
        raise ValueError('Expected independent canonical/reading comparison')
    if len(spec['cases'])!=10 or [c['comparison_arm'] for c in spec['cases']]!=['canonical']*5+['reading']*5:
        raise ValueError('Expected five fixed factual questions in each arm')
    first,second=spec['cases'][:5],spec['cases'][5:]
    if any(a['question']!=b['question'] or a['source_packet_ids']!=b['source_packet_ids'] or a['paired_case_id']!=b['paired_case_id'] for a,b in zip(first,second)):
        raise ValueError('Factual comparison sources/questions differ across arms')
    class PairedBot(GhostChat):
        def __init__(self,*args):super().__init__(*args);self.cases=iter(spec['cases'])
        def answer(self,question,generate,*,state=None,history=()):
            case=next(self.cases)
            if history or case['question']!=question:raise ValueError('Factual cases must be independent and ordered')
            if case['comparison_arm']=='reading':
                bot=SourceReadingChat(self.profile,self.collection,self.ledger);bot.retriever=self.retriever
                result=bot.answer(question,generate,state=state,history=history)
            else:result=super().answer(question,generate,state=state,history=history)
            return {**result,'comparison_arm':case['comparison_arm'],'paired_case_id':case['paired_case_id']}
    return journal_evaluate(profile,collection,ledger,spec,generate,output,identity,bot_factory=PairedBot)
