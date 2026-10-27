"""Inspect new proposition coverage with oracle IDs; never route model requests."""
import argparse
import json
from pathlib import Path
from build_ghost_candidate_probe import ROOT,sha
from build_reviewed_source_ledger import build
from audit_reviewed_source_coverage import INSPECTIONS
from reviewed_source_display import records_for_packets,render_selection

ADDITIONS={
    'hearing_fields':['hearing_corrected_march_date'],
    'eastman_theory_and_pressure':['eastman_initiative_and_tactics'],
    'blumkin_inference_scopes':['blumkin_disclosure_purpose'],
    'v2_report_routes_and_hypothesis':['rakovsky_transport_confidence','rakovsky_eventual_fate_boundary','bessedovsky_informant_boundary'],
    'v2_exposure_denominator':['ilo_female_gradient_shares','ilo_female_gradient_sum','ilo_local_estimate_boundary']}


def audit(root=ROOT):
    _,collection,ledger,manifest=build(root);raw=(root/'evaluation/ghost_smoke_v27/evaluation_spec.json').read_bytes();spec=json.loads(raw)
    if [c['id'] for c in spec['cases']]!=list(ADDITIONS):raise ValueError('Original question order differs')
    rows=[];outputs={}
    for case in spec['cases']:
        records=records_for_packets(collection,ledger,case['source_packet_ids']);old,gaps=INSPECTIONS[case['id']]
        ids=old+ADDITIONS[case['id']];shown=render_selection(ids,records);outputs[case['id']]=shown
        rows.append({'case_id':case['id'],'original_identified_gaps':gaps,'new_supporting_claim_ids':ADDITIONS[case['id']],
            'oracle_reviewed_claim_ids':ids,'rendered_claim_ids':shown['rendered_claim_ids'],'omitted_selected_claim_ids':shown['omitted_selected_claim_ids'],
            'response_fingerprint':shown['response_fingerprint'],'model_selection_measured':False,'task_accuracy_claimed':False})
    return {'status':'offline_proposition_representation_review','model_calls':0,'active_model_changed':False,'ledger_manifest':manifest,
        'evaluation_spec_sha256':sha(raw),'rows':rows,'review':'Unblinded implementing-agent source review, not independent or human validation',
        'interpretation':'Previously identified details have reviewed proposition coverage; selector relevance and displayed task completion remain unmeasured'},outputs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);args=p.parse_args();report,examples=audit()
    args.output.mkdir(parents=True,exist_ok=True)
    for name,value in [('coverage.json',report),('oracle_displays.json',examples)]:
        raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();path=args.output/name
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing extension review differs; use a new directory')
    for name,value in [('coverage.json',report),('oracle_displays.json',examples)]:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'claim_count':report['ledger_manifest']['original_claims_preserved']+len(report['ledger_manifest']['added_claim_ids']),
        'new_claims':len(report['ledger_manifest']['added_claim_ids']),'oracle_rendered_counts':[len(r['rendered_claim_ids']) for r in report['rows']],
        'model_calls':0,'model_selection_or_task_accuracy_measured':False},indent=2))
