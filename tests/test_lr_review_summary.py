"""Detect accidental pooling of fresh/regression quality or altered review evidence."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from scripts import summarize_lr_test_review as summary
class LRReviewTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.run=Path(self.temp.name)/'results';self.review=self.run/'review';self.review.mkdir(parents=True)
  self.source=[];self.keys=[];self.scores={1:[],2:[],3:[]}
  for i in range(48):
   category=list(summary.DIM)[i%24//6];split='fresh_diagnostic_holdout_v3' if i<24 else 'fresh_diagnostic_holdout_v2';shard=i//16+1
   for j,condition in enumerate(summary.CONDITIONS):
    rid=f'R{len(self.source)+1:03d}';response=f'Fixture answer {i} {j}';h=hashlib.sha256(response.encode()).hexdigest()
    self.source.append(dict(review_id=rid,case_id=f'case_{i:02d}',category=category,control_type='positive',split=split,response=response,token_count=12,finish_reason='stop',**dict.fromkeys(summary.SCORE_FIELDS,'')))
    self.keys.append(dict(review_id=rid,case_id=f'case_{i:02d}',condition=condition,shard=shard,response_sha256=h))
    main=([2,1,3] if i<24 else [2,1,0])[j]
    self.scores[shard].append(dict(review_id=rid,case_id=f'case_{i:02d}',category=category,response_sha256=h,**{d:(main if d==summary.DIM[category] else None) for d in summary.DIM.values()},critical_failure=0,critical_failure_types=[],review_notes='Fixture explicitly supports the tested dimension.'))
  self.write()
 def write(self):
  def jl(name,values):(self.review/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
  jl('review_rows.jsonl',self.source);jl('review_key.jsonl',self.keys)
  for i,values in self.scores.items():jl(f'agent_scores_shard_{i:02d}.jsonl',values)
  self.lock()
 def lock(self):
  names=['review_rows.jsonl','review_key.jsonl',*[f'agent_scores_shard_{i:02d}.jsonl' for i in range(1,4)]]
  lock=dict(reviewer_count=1,locked_before_unblinding=True,completed_utc='2026-10-25T00:00:00+00:00',files={n:hashlib.sha256((self.review/n).read_bytes()).hexdigest() for n in names})
  (self.review/'score_lock_before_unblinding.json').write_text(json.dumps(lock))
 def test_fresh_and_regressions_remain_separate(self):
  result=summary.build(self.run)
  self.assertNotIn('models',result)
  fresh=result['groups']['fresh_primary'];reg=result['groups']['development_regression'];new=summary.CONDITIONS[-1]
  self.assertEqual(fresh['models'][new]['category_scores']['chronology']['mean_main_score_0_4'],3)
  self.assertEqual(reg['models'][new]['category_scores']['chronology']['mean_main_score_0_4'],0)
  pair='_vs_'.join(summary.CONDITIONS[1:])
  self.assertEqual(fresh['paired'][pair]['score_improved'],24);self.assertEqual(reg['paired'][pair]['score_degraded'],24)
 def test_score_edit_after_lock_rejected(self):
  p=self.review/'agent_scores_shard_01.jsonl';p.write_text(p.read_text()+'\n')
  with self.assertRaisesRegex(ValueError,'score lock hash'):summary.build(self.run)
 def test_missing_lock_rejected(self):
  (self.review/'score_lock_before_unblinding.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'score lock'):summary.build(self.run)
 def test_missing_case_cannot_pass_after_relocking(self):
  self.scores[1].pop();self.write()
  with self.assertRaisesRegex(ValueError,'exactly 48'):summary.build(self.run)
 def test_split_relabelling_cannot_pass_after_relocking(self):
  self.source[0]['split']='fresh_diagnostic_holdout_v2';self.write()
  with self.assertRaisesRegex(ValueError,'differs across conditions'):summary.build(self.run)
if __name__=='__main__':unittest.main()
