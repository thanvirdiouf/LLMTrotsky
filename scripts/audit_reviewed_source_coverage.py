"""CPU-only representation/packing audit; oracle IDs are not a runtime router."""
import argparse
import json
from pathlib import Path
from build_ghost_candidate_probe import ROOT,sha
from build_ghost_chat import build
from reviewed_source_display import records_for_packets,render_selection
from ghost_chat import plan_response_budget
from grounded_chat import fingerprint

# Development inspection of the original five questions, not model selection
# or an answer key that may be supplied to a selector at inference time.
INSPECTIONS={
    'hearing_fields':(['hearing_correction','hearing_1902','hearing_secretary_count_limit'],
        ['Exact corrected March date is in source blocks but absent from these immutable statement forms.']),
    'eastman_theory_and_pressure':(['eastman_ultimatum','eastman_retrospect','eastman_sincerity_theory_distinction'],
        ['Eastman own initiative/risk is in source but not represented by these immutable statements.']),
    'blumkin_inference_scopes':(['blumkin_visitor_identification_limit','blumkin_disclosure','blumkin_allegation','blumkin_duty','blumkin_reform','blumkin_prediction'],[]),
    'v2_report_routes_and_hypothesis':(['rakovsky_reports','rakovsky_geography','rakovsky_limit','rakovsky_reasoning','bessedovsky_report','bessedovsky_limit'],
        ['Two-source author confidence in the Barnaul-to-Moscow move lacks a dedicated immutable statement.',
         'Eventual-fate and named-informant limits need explicit source-reviewed coverage; do not invent a later-outcome story.']),
    'v2_exposure_denominator':(['ilo_exposure_estimate','ilo_adoption_limit','ilo_transformation_assessment'],
        ['Female-employment gradient3/4 values and their10.4sum are in source but absent from these statements.',
         'Missing local composition/task/adoption requirements need explicit bounded coverage; no office count follows.'])}


def audit(root=ROOT):
    _,collection,ledger=build(root,source_plan='config/ghost_source_plan_v4.json')
    raw=(root/'evaluation/ghost_smoke_v27/evaluation_spec.json').read_bytes();spec=json.loads(raw)
    if [c['id'] for c in spec['cases']]!=list(INSPECTIONS):raise ValueError('Original question coverage differs')
    rows=[];outputs={}
    for case in spec['cases']:
        ids,gaps=INSPECTIONS[case['id']];records=records_for_packets(collection,ledger,case['source_packet_ids'])
        shown=render_selection(ids,records);selected={i:next(c for c in ledger['claims'] if c['id']==i) for i in ids}
        legacy,_,budget=plan_response_budget(case['question'],selected,{'_answer_mode':'facts','_detail':'normal'})
        rows.append({'case_id':case['id'],'source_packet_ids':case['source_packet_ids'],'oracle_reviewed_claim_ids':ids,
            'new_rendered_claim_ids':shown['rendered_claim_ids'],'legacy_budget_claim_ids':legacy,
            'legacy_hidden_before_budget_accounting':[i for i in ids if i not in legacy and i not in budget['omitted_fact_claim_ids']],
            'known_statement_representation_gaps':gaps,'rendered_response_fingerprint':shown['response_fingerprint'],
            'model_selection_performed':False,'completeness_measured_as_pass':False})
        outputs[case['id']]=shown
    return {'status':'offline_development_coverage_audit','source_plan':'config/ghost_source_plan_v4.json',
        'source_plan_sha256':sha((root/'config/ghost_source_plan_v4.json').read_bytes()),
        'collection_fingerprint':fingerprint(collection),'ledger_fingerprint':fingerprint(ledger),
        'renderer_sha256':sha((root/'scripts/reviewed_source_display.py').read_bytes()),
        'audit_code_sha256':sha((root/'scripts/audit_reviewed_source_coverage.py').read_bytes()),
        'active_pointer_sha256':sha((root/'config/active_chat.json').read_bytes()),
        'claim_count':len(ledger['claims']),'evaluation_spec_sha256':sha(raw),'model_calls':0,
        'oracle_ids_are_not_runtime_question_routing':True,'active_model_changed':False,'rows':rows},outputs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report,outputs=audit();args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'oracle_displays.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'model_calls':0,'claim_count':report['claim_count'],
        'cases_with_representation_gaps':sum(bool(r['known_statement_representation_gaps']) for r in report['rows']),
        'cases_with_hidden_packing_loss':sum(bool(r['legacy_hidden_before_budget_accounting']) for r in report['rows'])},indent=2))
