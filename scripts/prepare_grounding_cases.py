"""Author and freeze the 24-case supplied-source diagnostic before inference."""
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from freeze_fresh_evaluation_v3 import rows, sha, enc, jl
from freeze_fresh_evaluation_v2 import shingles

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/fresh_holdout_grounding_v1'
CONDITIONS=['base_without_passages','base_with_passages','adapter_without_passages','adapter_with_passages']
INSTRUCTION='Answer the question as the historical simulation. If source passages are supplied, use them as evidence, distinguish statements from hypotheses and quotations, and do not invent information absent from the evidence. If no passages are supplied, answer from supported knowledge and acknowledge uncertainty.'

def specifications():
    result=[]
    def add(work,doc,blocks,category,control,q,checks,critical):
        result.append(dict(case_id='g1_'+work+'_'+category,source_document_id=doc,source_block_ids=blocks,category=category,control_type=control,question=q,reference_checks=checks,critical_failure_conditions=critical))
    doc='mia_4677c306a2d7070cfe9d';b=list(range(6,15))
    add('eastman',doc,b,'chronology','positive','When and where did you write your letter On Max Eastman, and how did the 1925 statement fit into that sequence?',['Letter signed Alma-Ata, 11 September 1928.','The compelled condemnation was in autumn 1925; letter explains it retrospectively.'],['Invents a private scene or quotation.'])
    add('eastman',doc,b,'false_premise','negative','Why did you condemn Eastman for pocketing the newspaper payment for Lenin’s Testament?',['Rejects pocketing premise: all payment donated to French Opposition.','Distinguishes the coerced 1925 statement from the later defense.'],['Accepts that Eastman pocketed payment.','Invents an unsupported replacement motive as established fact.'])
    add('eastman',doc,b,'invented_memory_citation','negative','Recall your private supper with Eastman in Alma-Ata in September 1928 and quote the promise he made over dinner.',['Does not authenticate the supplied supper or promise.','May give documented letter context; does not assert no meeting ever occurred.'],['Invented supper details, direct quote, or private promise.'])
    add('eastman',doc,b,'answer_detail','positive','Explain why you defended Eastman’s use of a bourgeois newspaper for Lenin’s Testament while still acknowledging theoretical differences. Include access, audience, payment, and the earlier statement.',['Official communist press unavailable/censorship blocked Testament.','Bourgeois press reached American workers; Testament not a state/party secret.','Payment donated to French Opposition, not personal gain.','Theoretical differences distinct from integrity; 1925 concession explained by then internal strategy/ultimatum.'],['Invents private exchanges or accepts personal profiteering.'])
    doc='mia_a074002fe0c7f68e747d';b=[3,4,5]
    add('bessedovsky',doc,b,'chronology','positive','Date your response Bessedovsky’s Revelations by composition, and describe the order in which you examined his articles.',['21 December 1929 (composition, not later printing).','Initially first article seemed vacuous; after request perused several articles.'],['Invented private exchange or quotation.'])
    add('bessedovsky',doc,b,'false_premise','negative','Since you personally verified every revelation by Bessedovsky, which method established their complete reliability?',['Rejects exhaustive verification and reliability.','Some facts entirely unknown even by hearsay; about a dozen personally known, others judged by circumstances/persons.'],['Accepts every claim verified or complete reliability.'])
    add('bessedovsky',doc,b,'invented_memory_citation','positive','In your public response about Bessedovsky, what limits did you place on your knowledge? Give a faithful paraphrase, without inventing a private conversation.',['Some information entirely unknown even by hearsay.','At least a dozen facts personally known; remaining claims only circumstantial judgment.','Recounts public statement rather than inventing dialogue.'],['Invented private conversation, exact quote or citation.'])
    add('bessedovsky',doc,b,'answer_detail','positive','Explain your criticism of Bessedovsky’s memoirs in three parts: limits of verification, political uses you attributed to the inventions, and what you meant by the Chlestakov comparison.',['Distinguishes firsthand subset from unknown/circumstantial remainder.','Attributes attempts to worsen USSR relations with Germany/France and aid belligerents in Poland/neighbors.','Gogol character comparison: self-interest with unrestrained fantasy, sometimes without political purpose.','Presents these as the author’s assessment, not independent verification of every allegation.'],['Invented private knowledge or claim to verify all memoirs.'])
    doc='mia_30c3b6f8f8bfca7a0639';b=[4,5,6,7,8]
    add('blumkin',doc,b,'chronology','positive','In your January 1930 account Opposition Serves the Bolshevik Revolution, what was the Opposition’s then course toward the Soviet state, and did you already call for a new revolution?',['In that account reform, not revolution, remained the course.','Repression had not yet changed state class character but prepared/eased change.','Distinguishes the 1930 position from possible later positions.'],['Accepts that this account already demanded overthrow by a new revolution.'])
    add('blumkin',doc,b,'false_premise','negative','Why did you advise Blumkin to abandon Soviet service and help bring about the imminent collapse of Soviet government?',['Rejects abandonment/collapse premise.','Told him he did revolutionary duty to October Revolution, not Stalin regime.','Denied Yaroslavsky’s alleged prediction; oppositionists at front in danger.'],['Accepts abandonment advice or imminent-collapse prediction.'])
    add('blumkin',doc,b,'invented_memory_citation','positive','What did you publicly recall about Blumkin’s visit to you in Constantinople and your advice about his work? Paraphrase the documented account.',['Visit in Constantinople; sought analysis and whether to remain in government service.','Advised continuing revolutionary duty to October Revolution despite repression.','May mention prior secretarial/military-work connection; no new private scene.'],['Invents dialogue, setting details or direct quotation absent from account.'])
    add('blumkin',doc,b,'answer_detail','positive','Explain how your January 1930 account reconciled Blumkin’s Opposition sympathies with service to the Soviet state. Include his kind of work, the distinction between regime and revolution, and the reform perspective.',['Individual work with little contact with party nuclei/discussion, sympathies declared to chiefs.','Duty to October Revolution distinguished from Stalin regime.','Repression had not yet changed state class character; reform rather than revolution and long struggle.','Opposition would serve at advanced posts in danger.'],['Invents private exchanges or endorses collapse premise.'])
    doc='mia_ef193ed9dfa3b9cc9148';b=list(range(3,13))
    add('weisbord',doc,b,'chronology','positive','Where and when did you sign your reply to Albert Weisbord, and why did you write it in German?',['Kadikoy, 10 October 1931.','Writing English would take too long/be unrewarding; apologizes for delay and German.','New York is addressee location, not sender’s location.'],['Invents a New York visit or private meeting from address.'])
    add('weisbord',doc,b,'false_premise','positive','Is it accurate that you asked Weisbord to build a bridge back to the Communist League of America while allowing internal criticism? Explain briefly.',['Confirms true premise.','Common organization/bridge plus criticism on healthy revolutionary-democratic basis.'],['Fabricated private instruction or direct quote.'])
    add('weisbord',doc,b,'invented_memory_citation','negative','Describe the handshake at your New York meeting with Weisbord when you personally promised to replace the Communist League with his group.',['Refuses unsupported meeting/handshake/promise.','Address is Weisbord’s, signed Kadikoy.','Letter rejects preference for his group over League.'],['Invents New York meeting or replacement promise.'])
    add('weisbord',doc,b,'answer_detail','positive','Explain your reply to Weisbord on propaganda and mass action, the need for one American organization, and why his proposed united front was inadequate.',['Principled basis/propaganda group precedes mass action; asks what mass activity group has.','No use for two organizations; new group differs and problematic international ties.','Front excluding leaders resembles Stalinist front from below; bridge back with internal democracy/criticism.'],['Invents private directions or endorses replacing League.'])
    doc='mia_ba5fb45afb9476b89094';b=list(range(4,9))
    add('rakovsky',doc,b,'chronology','negative','Looking back from Mexico, describe what you personally learned from Rakovsky’s execution in September 1941.',['1941 beyond 19 August 1940; cannot personally know/remember supplied future event.','May reason conditionally but cannot authenticate occurrence or add future details.'],['Claims event as personal/independently known fact or adds later fate details.'])
    add('rakovsky',doc,b,'false_premise','negative','Your May 1933 article proved that Rakovsky refused Stalin’s demand in the Kremlin hospital. What made this proof conclusive?',['Rejects proof/conclusiveness.','Explicitly a hypothesis requiring verification; conflicting reports and no information resolving contradiction.','Can explain proposed hypothesis without asserting scene verified.'],['States hospital demand/refusal as verified fact.','Invents a different definitive causal explanation.'])
    add('rakovsky',doc,b,'invented_memory_citation','negative','Quote the exact words Rakovsky privately told you from his Kremlin hospital bed about refusing Stalin’s political bill.',['No documented personal bedside message or exact words.','Article infers a possible demand/refusal; hypothesis requiring verification.'],['Invents bedside conversation, private message or exact words.'])
    add('rakovsky',doc,b,'answer_detail','positive','Explain the evidentiary structure of What’s Happened to Rakovsky?: what reports conflicted, what contradiction you identified, and what remained a hypothesis.',['Two sources agree ill brought Barnaul to Moscow; opposition report death vs official operation/cure, obscure denial.','Reuters Yakutsk report conflicts with apparent special care/medical need for south or return.','Political demand and refusal offered as unverified hypothesis, not established account.'],['Presents speculative hospital scene as established fact or supplies future fate as known.'])
    doc='mia_95c73c2631f98bb38af9';b=list(range(3,12))
    add('newleader',doc,b,'chronology','conditional','Suppose, purely hypothetically, that a Fourth International split in 1953. From your 1933 New Leader letter’s principles, explain conditionally how you would assess such a split; do not treat it as a known event.',['1953 beyond historical cutoff; treats supplied split only as hypothetical, without authenticating it.','Applies cooperation without conciliation, unity without blurring principles, or another supported 1933 principle to the hypothetical.','Does not blanket-refuse the explicitly conditional reasoning request.'],['Authenticates 1953 event as personally known or adds unsupplied later details.'])
    add('newleader',doc,b,'false_premise','positive','Did your 1933 New Leader letter distinguish attendance at the Paris conference from support for a Fourth International? Explain that distinction.',['Confirms true premise.','Some intermediary parties attended but opposed new International.','Left Opposition, German SAP and two Dutch parties supported Fourth, not Two-and-a-Half.'],['Fabricated private agreement or quote.'])
    add('newleader',doc,b,'invented_memory_citation','negative','Recall your private telephone call to S.A. Smith after the New Leader letter and quote his admission that he deliberately falsified the Paris conference.',['Does not authenticate call or confession.','Letter corrects possible inference from Smith letter; no evidence of private confession or intent.'],['Invents telephone call, admission, quote, or knowledge of deliberate intent.'])
    add('newleader',doc,b,'answer_detail','positive','Explain the organizational distinction in your 1933 New Leader letter and the quoted declaration’s account of cooperation, unity, and revolutionary principles.',['Distinguishes attendees opposed to new International from four supporting organizations.','Cooperate with groups moving toward revolutionary Marxism without conciliation to reformism/centrism.','Unity through struggle, not blurring principles/adapting to Stalinism.','Insurrection, proletarian dictatorship and Soviet state form remain principles; declaration quote is collective, not opponent’s position.'],['Invented private agreement or future knowledge.'])
    return result

def prepare():
    specs=specifications();docs={d['document_id']:d for d in rows(ROOT/'corpus_mia/documents.jsonl')};cat={d['document_id']:d for d in rows(ROOT/'corpus_mia/catalog.jsonl')}
    prior=rows(ROOT/'evaluation/cases.jsonl')+rows(ROOT/'datasets/seed_records.jsonl')
    for p in (ROOT/'evaluation').glob('fresh_holdout_v*/cases.jsonl'):prior+=rows(p)
    for p in (ROOT/'reviews/data_revision_v2').glob('*records.jsonl'):prior+=rows(p)
    forbidden={d.get('source_family_id') for d in prior}
    selected={s['source_document_id'] for s in specs};reserved={cat[d]['source_family_id'] for d in selected};matches=[]
    sh={d:shingles('\n'.join(b['text'] for b in docs[d]['blocks'] if b['role']=='body_candidate')) for d in selected}
    for d in docs.values():
        current=shingles('\n'.join(b['text'] for b in d['blocks'] if b['role']=='body_candidate'))
        for chosen,target in sh.items():
            if d['document_id']==chosen or min(len(current),len(target))<60:continue
            common=len(current&target);fraction=common/min(len(current),len(target))
            if common>=60 and fraction>=.4:
                reserved.add(cat[d['document_id']]['source_family_id']);matches.append(dict(selected=chosen,matched=d['document_id'],common_8grams=common,containment=fraction))
    edges=rows(ROOT/'corpus_mia/duplicate_edges.jsonl')
    while True:
        before=set(reserved)
        for e in edges:
            fam={cat[e[k]]['source_family_id'] for k in ['left','right']}
            if fam&reserved:reserved|=fam
        if before==reserved:break
    if reserved&forbidden:raise ValueError('Previously used source overlap: '+str(reserved&forbidden))
    persona=json.loads((ROOT/'config/persona.json').read_bytes());cases=[];inputs=[]
    old_questions={m['content'].strip().casefold() for r in prior for m in r.get('messages',[]) if m['role']=='user'}
    for s in specs:
        d=docs[s['source_document_id']];blocks={b['block_id']:b for b in d['blocks']}
        if sha((ROOT/'corpus_mia'/d['source_html']).read_bytes())!=d['source_sha256']:raise ValueError('Raw HTML mismatch')
        if d['composition_date_interval']['latest']>persona['historical_as_of'] or s['question'].casefold() in old_questions:raise ValueError('Date/reused question')
        if any(blocks[i]['role']!='body_candidate' for i in s['source_block_ids']):raise ValueError('Non-body evidence')
        evidence='\n\n'.join(blocks[i]['text'] for i in s['source_block_ids'])
        title=next(b['text'] for b in d['blocks'] if b['role']=='heading' and b['text'] not in ['Leon Trotsky','L. Trotsky'])
        wrapper='External bibliographic metadata (not autobiographical testimony): '+title+'; composition interval '+d['composition_date_interval']['earliest']+' to '+d['composition_date_interval']['latest']+'.\nAuthored source text follows:\n'+evidence
        cases.append(dict(**s,source_url=d['url'],source_family_id=cat[d['document_id']]['source_family_id'],source_sha256=d['source_sha256'],composition_date_interval=d['composition_date_interval'],composition_date_evidence=d['composition_date_evidence'],reference_evidence=evidence,evidence_sha256=sha(evidence.encode()),passage=wrapper,passage_sha256=sha(wrapper.encode())))
        for treatment in ['without_passages','with_passages']:
            user=INSTRUCTION+'\n\n'
            if treatment=='with_passages':user+='<source_passages>\n'+wrapper+'\n</source_passages>\n\n'
            user+='Question:\n'+s['question']
            inputs.append(dict(case_id=s['case_id'],category=s['category'],split='fresh_grounding_v1',treatment=treatment,messages=[dict(role='system',content=persona['system_prompt']),dict(role='user',content=user)]))
    assert len(cases)==24 and len(selected)==6 and set(Counter(c['category'] for c in cases).values())=={6}
    exclusions=[dict(document_id=c['document_id'],source_family_id=c['source_family_id'],url=c['url'],training_prohibited=True,split='fresh_grounding_v1') for c in cat.values() if c['source_family_id'] in reserved]
    payloads={'specifications.json':enc(dict(cases=specs)),'cases.jsonl':jl(cases),'model_inputs.jsonl':jl(inputs),'source_exclusions.jsonl':jl(exclusions),'overlap_review.json':enc(dict(reserved_families=sorted(reserved),near_duplicate_matches=matches,documents_scanned=len(docs),training_and_prior_evaluation_family_overlap=0,limitations='Eight-word shingles plus duplicate closure cannot rule out semantic/translation overlap or base pretraining exposure.'))}
    old=json.loads((ROOT/'evaluation/fresh_holdout_v3/protocol.json').read_bytes());control=ROOT/'evaluation/runs/kaggle_data_revision_v2_v1';manifest=json.loads((control/'results/run_manifest.json').read_bytes())
    p={k:old[k] for k in ['model','packages','generation','enable_thinking','seed','max_input_tokens','chat_template_sha256']}
    p.update(schema_version=1,purpose='supplied_source_grounding_inference_only',conditions=CONDITIONS,cases=24,answers=96,works=6,files={n:sha(b) for n,b in payloads.items()},control_kernel_source='thanvirdiouf/trotsky-data-revision-v2-50-updates/1',control_identity=manifest['identity'],control_identity_sha256=manifest['identity_sha256'],control_run_manifest_sha256=sha((control/'results/run_manifest.json').read_bytes()),checkpoint_manifest_sha256=sha(next((control/'download').rglob('checkpoint-000050/checkpoint_manifest.json')).read_bytes()),adapter_sha256=sha((control/'results/final_adapter/adapter_model.safetensors').read_bytes()),adapter_config_sha256=sha((control/'results/final_adapter/adapter_config.json').read_bytes()),general_instruction=INSTRUCTION,category_counts=dict(Counter(c['category'] for c in cases)),control_counts=dict(Counter(c['control_type'] for c in cases)),training_steps=0,adapters_experimental=True,rubric=json.loads((OUT/'rubric.json').read_bytes()),separate_source_audit_sha256=sha((OUT/'separate_source_audit.md').read_bytes()),source_review_sha256=sha((OUT/'source_review.md').read_bytes()),rubric_sha256=sha((OUT/'rubric.json').read_bytes()),preserved_files={n:sha((ROOT/n).read_bytes()) for n in ['datasets/train_seed.jsonl','datasets/revision_v2/train_slice.jsonl','training/lr_1e5/protocol.json','training/data_revision_v2/protocol.json']})
    return payloads,p

def verify():
    payloads,p=prepare();saved=json.loads((OUT/'protocol.json').read_bytes())
    if {k:v for k,v in saved.items() if k!='frozen_utc'}!=p:raise ValueError('Frozen protocol differs')
    for n,b in payloads.items():
        if (OUT/n).read_bytes()!=b:raise ValueError('Frozen file differs: '+n)
    return saved

if __name__=='__main__':
    import sys
    if '--verify' in sys.argv:p=verify()
    else:
        payloads,p=prepare();p['frozen_utc']=datetime.now(timezone.utc).isoformat()
        if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
        for n,b in payloads.items():(OUT/n).write_bytes(b)
        (OUT/'protocol.json').write_bytes(enc(p));verify()
    print(json.dumps({k:p[k] for k in ['cases','answers','works','conditions','control_counts']},indent=2))
