"""Small present-day ghost smoke; journal every stage and displayed answer."""
import json
from pathlib import Path
try:
    from .grounded_chat import fingerprint
    from .ghost_chat import GhostChat,initial_state
    from .evaluate_grounded_chat_v2 import read_journal
    from .kaggle_grounding import atomic_journal
except ImportError:
    from grounded_chat import fingerprint
    from ghost_chat import GhostChat,initial_state
    from evaluate_grounded_chat_v2 import read_journal
    from kaggle_grounding import atomic_journal


def evaluate(profile,collection,ledger,spec,generate,output,identity):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    call_path,answer_path=output/'model_calls.jsonl',output/'answers.jsonl'
    old_calls,old_answers=read_journal(call_path),read_journal(answer_path)
    calls,answers=[],[]
    bot=GhostChat(profile,collection,ledger)
    jobs=[(c['id'],[c['question']]) for c in spec['cases']]+[('ghost_conversation',spec['conversation'])]
    for case_id,questions in jobs:
        history=[];state=initial_state(collection,ledger)
        for index,question in enumerate(questions,1):
            key=f'{case_id}/{index}';step=0
            def cached(messages):
                nonlocal step
                step+=1
                signature=fingerprint({'identity':identity,'key':key,'step':step,'messages':messages})
                if len(calls)<len(old_calls):
                    row=old_calls[len(calls)]
                    if row.get('signature')!=signature or row.get('messages')!=messages or row.get('output_sha256')!=fingerprint(row.get('output')):
                        raise ValueError('Smoke call resume identity differs')
                else:
                    result=generate(messages)
                    row={'key':key,'step':step,'signature':signature,'messages':messages,'output':result,'output_sha256':fingerprint(result)}
                calls.append(row)
                if len(calls)>len(old_calls):atomic_journal(call_path,calls)
                return row['output']
            result=bot.answer(question,cached,state=state,history=history)
            record={**result,'key':key,'case_id':case_id,'turn':index,'model_calls':step,'identity_sha256':fingerprint(identity)}
            if len(answers)<len(old_answers) and old_answers[len(answers)]!=record:
                raise ValueError('Smoke answer or state replay differs')
            answers.append(record)
            if len(answers)>len(old_answers):atomic_journal(answer_path,answers)
            state=result['state_after'];history += [{'role':'user','content':question},{'role':'assistant','content':result['response']}]
            print(key,result['status'],flush=True)
    if len(old_calls)>len(calls) or len(old_answers)>len(answers):raise ValueError('Unexpected smoke journal tail')
    return answers,calls
