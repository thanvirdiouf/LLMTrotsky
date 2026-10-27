import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_hardware_probe import build,sha
from verify_ghost_hardware_probe import verify
from ghost_hardware_probe import probe

class HardwareProbeTest(unittest.TestCase):
    def fixture(self,root):
        script=root/'scripts/ghost_hardware_probe.py';script.parent.mkdir(parents=True);script.write_bytes((ROOT/'scripts/ghost_hardware_probe.py').read_bytes())
        metadata=root/'training/kaggle_cli/grounded_chat_v1/kernel-metadata.json';metadata.parent.mkdir(parents=True)
        metadata.write_bytes((ROOT/'training/kaggle_cli/grounded_chat_v1/kernel-metadata.json').read_bytes())
        stage=build(root=root);download=root/'evaluation/runs/ghost_hardware_v1/download'
        code=download/'ghost_hardware_bundle/ghost_hardware_probe.py';code.parent.mkdir(parents=True);code.write_bytes(script.read_bytes())
        memory=16*1024**3
        result={'schema_version':1,'purpose':'hardware_inventory_only','probe_source_sha256':sha(script.read_bytes()),'model_weights_loaded':False,
            'model_calls':0,'training_steps':0,'adapters_loaded':[],'gpu_count':2,'devices':[{'index':i,'name':'Tesla T4','total_memory_bytes':memory,'reported_total_bytes':memory,'free_memory_bytes':memory-1024**3} for i in range(2)],
            'ram_bytes':{'MemTotal':32*1024**3,'MemAvailable':24*1024**3},'working_disk_bytes':{'total':100*1024**3,'free':80*1024**3}}
        path=download/'ghost_hardware_results/result.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(result))
        return stage,script,path,result

    def test_probe_and_notebook_are_inventory_only_and_immutable(self):
        tree=ast.parse((ROOT/'scripts/ghost_hardware_probe.py').read_text())
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls&{'from_pretrained','snapshot_download','generate','backward','step','get_peft_model'})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);stage,script,_,_=self.fixture(root)
            metadata=json.loads((stage/'kernel-metadata.json').read_text());self.assertTrue(metadata['is_private']);self.assertFalse(metadata['enable_internet'])
            self.assertEqual(build(root=root),stage)
            script.write_text(script.read_text()+'\n# Changed input\n')
            with self.assertRaises(ValueError):build(root=root)
        with self.assertRaises(ValueError):build('../escape')

    def test_exact_source_and_consistent_inventory_required_without_claiming_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);stage,script,path,result=self.fixture(root)
            report=verify(root=root);self.assertEqual(report['gpu_count'],2);self.assertFalse(report['larger_model_fit_established'])
            for variant in ('count','memory','model_calls','loaded_weights','wrong_gpu'):
                bad=copy.deepcopy(result)
                if variant=='count':bad['gpu_count']=1
                elif variant=='memory':bad['devices'][0]['free_memory_bytes']=bad['devices'][0]['total_memory_bytes']+1
                elif variant=='model_calls':bad['model_calls']=1
                elif variant=='loaded_weights':bad['model_weights_loaded']=True
                else:bad['devices'][0]['name']='Unapproved device'
                path.write_text(json.dumps(bad))
                with self.subTest(variant=variant),self.assertRaises(ValueError):verify(root=root)
            path.write_text(json.dumps(result))
            (path.parent.parent/'ghost_hardware_bundle/ghost_hardware_probe.py').write_text('changed source')
            with self.assertRaises(ValueError):verify(root=root)

    def test_local_execution_fails_before_torch_import_or_output(self):
        with patch('ghost_hardware_probe.Path.is_dir',return_value=False),self.assertRaises(RuntimeError):probe('/tmp/not-created-ghost-probe.json')

    def test_runtime_with_simulated_devices_reports_only_hardware_and_no_model_calls(self):
        fake=SimpleNamespace(__version__='test',version=SimpleNamespace(cuda='test'),cuda=SimpleNamespace(
            device_count=lambda:2,get_device_properties=lambda i:SimpleNamespace(name='Tesla T4',total_memory=100,major=7,minor=5),
            mem_get_info=lambda i:(90,100)))
        def paths(value):return SimpleNamespace(is_dir=lambda:True) if str(value)=='/kaggle/working' else Path(value)
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,torch=fake),patch('ghost_hardware_probe.Path',side_effect=paths),patch('ghost_hardware_probe.shutil.disk_usage',return_value=SimpleNamespace(total=100,free=80)):
            output=Path(temp)/'result.json';result=probe(output)
            self.assertEqual(json.loads(output.read_text()),result);self.assertEqual(result['gpu_count'],2)
            self.assertEqual(result['model_calls'],0);self.assertFalse(result['model_weights_loaded'])
            self.assertEqual(result['probe_source_sha256'],sha((ROOT/'scripts/ghost_hardware_probe.py').read_bytes()))

if __name__=='__main__':unittest.main()
