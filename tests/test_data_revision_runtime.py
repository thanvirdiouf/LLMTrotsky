"""CPU guards for the data-only fixed-50-update runtime; no model or GPU is loaded."""
import builtins
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import symtable
import unittest
from unittest.mock import patch

from scripts import data_revision_protocol as protocol
from scripts import kaggle_data_revision_pilot as runner

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    pilot=json.loads((ROOT/'training/runs/kaggle_361474126/run_manifest.json').read_bytes())
    original_lock=json.loads((ROOT/'training/pilot_lock.json').read_bytes())
    system=json.loads((ROOT/'config/persona.json').read_bytes())['system_prompt']
    train=[{'example_id':f'fixture_{i:02d}','messages':[{'role':'system','content':system},
            {'role':'user','content':f'Fixture training question {i}?'},
            {'role':'assistant','content':f'Fixture answer {i}.'}]} for i in range(50)]
    train_bytes=''.join(json.dumps(r)+'\n' for r in train).encode()
    lock={'schema_version':1,'dataset':{'records':50,'sha256':protocol.digest(train_bytes),
            'example_sha256':{r['example_id']:protocol.fingerprint(r) for r in train}},
          'persona':original_lock['persona'],
          'contamination_guard':{'text_sha256':[],'span_sha256':[],'span_words':16}}
    categories=['chronology','false_premise','invented_memory_citation','answer_detail']
    evaluation=[{'case_id':f'fixture_case_{i:02d}','category':categories[i%4],
                 'split':'fresh_diagnostic_holdout_v2','messages':[{'role':'system','content':system},
                 {'role':'user','content':f'Fixture evaluation question {i}?'}]} for i in range(24)]
    eval_bytes=''.join(json.dumps(r)+'\n' for r in evaluation).encode()
    settings=deepcopy(pilot['identity']['settings']);settings['steps']=50
    p={'settings':settings,'conditions':protocol.CONDITIONS[:],
       'packages':pilot['identity']['packages'],'pilot_identity':pilot['identity'],
       'pilot_identity_sha256':pilot['identity_sha256'],
       'chat_template_sha256':pilot['identity']['chat_template_sha256'],
       'training':{'sha256':protocol.digest(train_bytes)},
       'evaluation':{'sha256':protocol.digest(eval_bytes),'generation':deepcopy(protocol.GENERATION)},
       'training_lock_fingerprint':protocol.fingerprint(lock)}
    return train_bytes,eval_bytes,lock,p,pilot


def metrics():
    return [{'step':i,'verified_optimizer_step':i,'example_id':f'fixture_{i-1:02d}',
             'finite_gradients':True,'frozen_gradients_absent':True,
             'input_tokens':300,'supervised_tokens':50,'loss':2.0,
             'gradient_norm_before_clip':1.5,'grad_scaler_scale':1.0} for i in range(1,51)]


class DataRevisionProtocolTests(unittest.TestCase):
    def test_exact_fifty_reviewed_rows_and_twenty_four_reference_free_inputs_pass(self):
        data,ev,lock,p,pilot=fixture()
        train,evaluation=protocol.validate_payloads(data,ev,lock,p,pilot)
        self.assertEqual((len(train),len(evaluation)),(50,24))

    def test_optimizer_configuration_cannot_change_silently(self):
        _,_,_,p,pilot=fixture()
        for key,value in [('learning_rate',2e-5),('rank',4),('epochs',2),('steps',51),
                          ('micro_batch_size',2),('amp_initial_scale',65536.0),('packing',True),('seed',43)]:
            altered=deepcopy(p['settings']);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                protocol.validate_settings(altered,pilot['identity']['settings'])
        altered=deepcopy(p['settings']);altered['steps']=50.0
        with self.assertRaises(ValueError):protocol.validate_settings(altered,pilot['identity']['settings'])

    def test_incomplete_or_modified_original_pilot_is_rejected(self):
        args=fixture()
        for mutation in ['status','steps','identity','check']:
            data,ev,lock,p,pilot=deepcopy(args)
            if mutation=='status':pilot['status']='failed'
            if mutation=='steps':pilot['completed_steps']=499
            if mutation=='identity':pilot['identity']['settings']['learning_rate']=2e-5
            if mutation=='check':pilot['checks']['resumable_checkpoint_round_trip']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                protocol.validate_payloads(data,ev,lock,p,pilot)

    def test_training_byte_hash_and_review_lock_bind_the_payload(self):
        data,ev,lock,p,pilot=fixture()
        with self.assertRaisesRegex(ValueError,'checksum'):
            protocol.validate_payloads(data+b'\n',ev,lock,p,pilot)
        altered=deepcopy(lock);altered['dataset']['example_sha256']['fixture_00']='wrong'
        with self.assertRaisesRegex(ValueError,'review lock'):
            protocol.validate_payloads(data,ev,altered,p,pilot)

    def test_reference_injection_malformed_content_duplicate_ids_and_decoding_are_rejected(self):
        for mutation in ['reference','duplicate','assistant','nontext','generation']:
            data,ev,lock,p,pilot=fixture();rs=[json.loads(s) for s in ev.splitlines()]
            if mutation=='reference':rs[0]['reference_evidence']='Do not send a reference target.'
            if mutation=='duplicate':rs[1]['case_id']=rs[0]['case_id']
            if mutation=='assistant':rs[0]['messages'][1]['role']='assistant'
            if mutation=='nontext':rs[0]['messages'][1]['content']=7
            if mutation=='generation':p['evaluation']['generation']['eos_token_id']=248044
            payload=''.join(json.dumps(r)+'\n' for r in rs).encode()
            p['evaluation']['sha256']=protocol.digest(payload)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                protocol.validate_payloads(data,payload,lock,p,pilot)

    def test_order_is_longest_first_then_unique_reproducible_seeded_order(self):
        encoded=[{'example_id':f'fixture_{i:02d}','dropped':False,'token_count':300+i} for i in range(50)]
        before=random.getstate()
        order=protocol.build_order(encoded)
        self.assertEqual(order[0],49)
        self.assertEqual(sorted(order),list(range(50)))
        self.assertEqual(order,protocol.build_order(encoded))
        self.assertEqual(before,random.getstate())
        for mutation in ['duplicate','overflow','invalid_tokens','bad_seed']:
            values=deepcopy(encoded)
            if mutation=='duplicate':values[1]['example_id']=values[0]['example_id']
            if mutation=='overflow':values[0]['dropped']=True
            if mutation=='invalid_tokens':values[0]['token_count']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                protocol.build_order(values,43 if mutation=='bad_seed' else 42)

    def test_fifty_actual_counters_and_finite_gradient_evidence_are_required(self):
        good=metrics();order=[r['example_id'] for r in good]
        self.assertTrue(protocol.verify_metrics(good,order))
        for mutation in ['skip','extra','reorder','nonfinite','frozen','zero_gradient','bad_tokens']:
            rs=deepcopy(good)
            if mutation=='skip':rs[12]['verified_optimizer_step']=12
            if mutation=='extra':rs.append(deepcopy(rs[-1]))
            if mutation=='reorder':rs[0],rs[1]=rs[1],rs[0]
            if mutation=='nonfinite':rs[3]['loss']=float('nan')
            if mutation=='frozen':rs[3]['frozen_gradients_absent']=False
            if mutation=='zero_gradient':rs[3]['gradient_norm_before_clip']=0.0
            if mutation=='bad_tokens':rs[3]['supervised_tokens']=301
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                protocol.verify_metrics(rs,order)


class DataRevisionRuntimeTests(unittest.TestCase):
    def test_local_gpu_work_is_refused_before_gpu_imports_or_file_reads(self):
        imported=[];original_import=builtins.__import__
        def checked_import(name,*args,**kwargs):
            imported.append(name)
            if name in {'torch','bitsandbytes','transformers','peft'}:
                raise AssertionError('GPU library imported before local guard')
            return original_import(name,*args,**kwargs)
        with patch.object(runner.Path,'is_dir',return_value=False),patch('builtins.__import__',side_effect=checked_import):
            with self.assertRaisesRegex(RuntimeError,'Kaggle'):
                runner.run_data_revision_pilot('/missing','/missing')
        self.assertNotIn('torch',imported)

    def test_new_portable_modules_compile_without_unbound_global_names(self):
        for name in ['data_revision_protocol.py','kaggle_data_revision_pilot.py']:
            text=(ROOT/'scripts'/name).read_text();compile(text,name,'exec')
            top=symtable.symtable(text,name,'exec')
            defined={s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
            scopes=[top]
            while scopes:
                scope=scopes.pop();scopes.extend(scope.get_children())
                for symbol in scope.get_symbols():
                    if symbol.is_referenced() and symbol.is_global():
                        with self.subTest(file=name,scope=scope.get_name(),name=symbol.get_name()):
                            self.assertTrue(symbol.get_name() in defined or hasattr(builtins,symbol.get_name()) or symbol.get_name()=='__file__')

if __name__=='__main__':unittest.main()
