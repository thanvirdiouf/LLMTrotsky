import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_reviewed_source_ledger import extend,save
from build_ghost_chat import historical_claim_supplements
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError
from reviewed_source_display import records_for_packets,render_selection


class ReviewedLedgerTests(unittest.TestCase):
    def fixture(self):
        historical={'id':'h','author':'Author','title':'H','url':'https://example.org/h','domain':'historical_record',
            'composition_date_interval':{'earliest':'1930-01-05','latest':'1930-01-05'},'composition_date_evidence':[],
            'context_boundary':'Entire bounded excerpt','blocks':[{'block_id':1,'speaker':'Author','role':'body_candidate','text':'First 1887, corrected to 1897.'}]}
        modern={'id':'m','author':'Institution','title':'M','url':'https://example.org/m','domain':'contemporary_evidence',
            'composition_date_interval':{'earliest':'2025-05-20','latest':'2025-05-20'},'composition_date_evidence':[],
            'observed_date':'2026-10-26','context_boundary':'Entire bounded snapshot',
            'blocks':[{'block_id':1,'speaker':'Institution','role':'research_summary','text':'5.7% of female employment in category3 and 4.7% in category4.'},
                {'block_id':2,'speaker':'Institution','role':'research_summary','text':'Potential exposure, not measured losses.'}]}
        collection={'packets':[historical,modern],'packets_sha256':fingerprint([historical,modern])}
        old={'id':'old','packet_id':'h','statement':'The source records a correction.','status':'documented_statement','attribution':'Author',
            'composition_date_interval':historical['composition_date_interval'],'domain':'historical_record',
            'anchors':[{'block_id':1,'speaker':'Author','text':historical['blocks'][0]['text']}]}
        ledger={'collection_sha256':collection['packets_sha256'],'claims':[old],'topic_aliases':{'h':[],'m':[]},'review':'Old review','reflection_basis_claim_ids':['old']}
        common={'status':'inference','review_note':'Unblinded development source review; no independent validation','attribution':'Application inference from the bounded source'}
        h={'id':'new_h','statement':'The source corrects1887to1897.','status':'documented_statement','attribution':'Recorded author correction',
            'anchors':copy.deepcopy(old['anchors']),'review_kind':'source_paraphrase','review_note':'Both years occur in the same answer.'}
        anchor={'block_id':1,'speaker':'Institution','text':modern['blocks'][0]['text']}
        arithmetic={**common,'id':'sum','statement':'Application sum is10.4percentage points, not observed losses.','anchors':[anchor],
            'review_kind':'arithmetic_derivation','derivation':{'kind':'add_percentage_shares','anchor_index':0,'operands':['5.7','4.7'],'result':'10.4',
                'denominator_phrase':'female employment','result_unit':'percentage_points'}}
        bound={**common,'id':'limit','statement':'No local workplace count is supplied.','anchors':[copy.deepcopy(anchor)],'review_kind':'bounded_absence','scope_block_ids':[1,2]}
        supplements={'schema_version':1,'review':'New unblinded source review','sources':[
            {'packet_id':'h','packet_fingerprint':fingerprint(historical),'claims':[h]},
            {'packet_id':'m','packet_fingerprint':fingerprint(modern),'claims':[arithmetic,bound]}]}
        return collection,ledger,supplements

    def test_mixed_domain_additions_preserve_original_records_and_authoritative_metadata(self):
        c,l,s=self.fixture();before=copy.deepcopy((c,l,s));new=extend(c,l,s)
        self.assertEqual((c,l,s),before);self.assertEqual(new['claims'][:1],l['claims'])
        self.assertEqual(new['reflection_basis_claim_ids'],l['reflection_basis_claim_ids']);self.assertEqual(new['topic_aliases'],l['topic_aliases'])
        records=records_for_packets(c,new,['h','m']);self.assertEqual(records['sum']['derivation']['result'],'10.4')
        self.assertEqual(records['sum']['status'],'inference');self.assertEqual(records['sum']['source']['composition_date_interval']['earliest'],'2025-05-20')
        self.assertEqual(records['sum']['source']['observed_date'],'2026-10-26')
        shown=render_selection(['new_h','sum','limit'],records)
        self.assertFalse(shown['semantic_guarantee']);self.assertEqual(shown['memory_updates'],[])

    def test_arithmetic_checks_value_tokens_denominator_and_inference_label(self):
        for change in ('result','operand','denominator','nonfinite','source_status'):
            c,l,s=self.fixture();claim=s['sources'][1]['claims'][0]
            if change=='result':claim['derivation']['result']='10.5'
            elif change=='operand':claim['derivation']['operands'][0]='6.7'
            elif change=='denominator':claim['derivation']['denominator_phrase']='all global job losses'
            elif change=='nonfinite':claim['derivation']['result']='NaN'
            else:claim['status']='documented_statement'
            with self.subTest(change=change),self.assertRaises(EvidenceError):extend(c,l,s)

    def test_bounded_absence_requires_complete_packet_scope_and_inference_status(self):
        for change in ('scope','status','speaker','metadata_override','duplicate','packet'):
            c,l,s=self.fixture();claim=s['sources'][1]['claims'][1]
            if change=='scope':claim['scope_block_ids']=[1]
            elif change=='status':claim['status']='explicit_limit'
            elif change=='speaker':claim['anchors'][0]['speaker']='Wrong person'
            elif change=='metadata_override':claim['composition_date_interval']={'earliest':'1940-01-01'}
            elif change=='duplicate':claim['id']='old'
            else:s['sources'][1]['packet_fingerprint']='changed'
            with self.subTest(change=change),self.assertRaises(EvidenceError):extend(c,l,s)

    def test_historical_supplement_guard_still_rejects_modern_packet(self):
        c,l,s=self.fixture();modern=copy.deepcopy(s['sources'][1])
        modern['claims']=[{k:v for k,v in modern['claims'][0].items() if k in ('id','statement','status','attribution','anchors')}]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'supplement.json';p.write_text(json.dumps({'schema_version':1,'review':'fixture','sources':[modern]}))
            with self.assertRaises(ValueError):historical_claim_supplements(c['packets'],l['claims'],p)

    def test_save_refuses_different_existing_files_before_any_writes(self):
        c,l,s=self.fixture();new=extend(c,l,s)
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);old=d/'ledger.json';old.write_text('frozen original')
            with self.assertRaises(EvidenceError):save(d,{},c,new,{})
            self.assertEqual(old.read_text(),'frozen original');self.assertFalse((d/'collection.json').exists())


if __name__=='__main__':unittest.main()
