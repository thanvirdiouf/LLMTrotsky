import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from generated_audit_design import assemble_spec,requests,replay,pair_groups
from build_ghost_candidate_probe import sha
from build_ghost_smoke import model_cases
from verify_ghost_candidate_audit import score_rows


class GeneratedAuditTests(unittest.TestCase):
    def fixture(self):
        baseline=json.loads((ROOT/'config/ghost_neutral_source_calibration_v1.json').read_text())
        design=json.loads((ROOT/'config/ghost_generated_audit_design_v1.json').read_text())
        text='Private generated paragraph placeholder for the wrong field.'
        design['generated_pair']['text_sha256']=sha(text.encode())
        source=[{'case_id':design['generated_pair']['case_id'],'segments':[{'id':design['generated_pair']['segment_id'],'text':text}]}]
        spec=assemble_spec(design,source,baseline)
        names=('source_audit.py','ghost_chat.py','grounded_chat.py','grounded_chat_v2.py')
        values={n:(ROOT/'scripts'/n).read_bytes() for n in names}
        packets=[{'id':pid,'title':'Fixture source','author':'Author','domain':'historical_record',
            'composition_date_interval':{'earliest':'1930-01-01','latest':'1930-01-01'},'composition_date_evidence':[],
            'context_boundary':'Fixture excerpt only','blocks':[{'block_id':1,'speaker':'Author','role':'body_candidate','text':'Fixture evidence.'}]} for pid in design['source_packet_ids']]
        values.update({'collection.json':json.dumps({'packets':packets}).encode(),'profile.json':b'{"reference_date":"2026-10-26"}'})
        return design,source,baseline,spec,values

    def test_private_candidate_hash_and_three_original_pairs_are_preserved(self):
        design,answers,baseline,spec,_=self.fixture()
        self.assertEqual(len(spec['cases']),10);self.assertEqual(len(spec['reviewer_only_pairs']),5)
        self.assertEqual(sum(c['expected_approved'] for c in spec['cases']),5)
        byid={c['id']:c for c in baseline['cases']}
        for case in spec['cases'][4:]:self.assertEqual(case,byid[case['id']])
        self.assertEqual(spec['cases'][0]['segments'][0]['text'],answers[0]['segments'][0]['text'])
        self.assertNotIn(answers[0]['segments'][0]['text'],json.dumps(design))
        self.assertNotIn('expected_approved',json.dumps(model_cases(spec)))
        changed=copy.deepcopy(answers);changed[0]['segments'][0]['text']='modified'
        with self.assertRaises(ValueError):assemble_spec(design,changed,baseline)

    def test_frozen_neutral_requests_have_same_three_sources_and_no_expected_labels(self):
        _,_,_,spec,values=self.fixture();jobs=requests(values,spec)
        self.assertEqual(len(jobs),10);systems=set()
        for case,job in zip(spec['cases'],jobs):
            self.assertEqual(case['id'],job['case_id']);data=json.loads(job['messages'][1]['content'])
            self.assertEqual(data['candidate_segments'],case['segments']);self.assertEqual(data['current_user_request'],case['question'])
            self.assertEqual([p['id'] for p in data['source_packets']],['hearing','rakovsky','bessedovsky'])
            self.assertNotIn('expected_approved',json.dumps(job['messages']));self.assertNotIn('local_rationale',json.dumps(job['messages']))
            systems.add(job['messages'][0]['content'])
        self.assertEqual(len(systems),1)

    def test_archived_parser_retains_blocks_and_empty_verdicts_as_distinct_outcomes(self):
        _,_,_,spec,values=self.fixture();rows=[]
        for case in spec['cases']:
            yes=case['expected_approved'];verdict={'id':'s1',**{k:yes for k in ('supported','certainty_preserved','attribution_preserved','temporal_safe','relevant','no_extra_claims')},'reason':'Specific comparison'}
            rows.append({'case_id':case['id'],'output':{'finish_reason':'stop','response':json.dumps({'verdicts':[verdict]})}})
        scored=score_rows(spec,replay(values,spec,rows));self.assertEqual(scored['pairs_both_correct'],5)
        rows[0]['output']['finish_reason']='length';rows[1]['output']['response']='{"verdicts":[]}'
        scored=score_rows(spec,replay(values,spec,rows))
        self.assertEqual(scored['counts']['blocked'],1);self.assertEqual(scored['counts']['unscored_judgment'],1)
        self.assertEqual(scored['pairs_both_correct'],4)
        groups=pair_groups(scored)
        self.assertEqual(groups['new_hearing_failures']['pairs_both_correct'],1)
        self.assertEqual(groups['familiar_controls_with_added_context']['pairs_both_correct'],3)


if __name__=='__main__':unittest.main()
