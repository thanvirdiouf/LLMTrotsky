"""Freeze a hardware-only notebook on the existing private free Kaggle target."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]


def sha(raw):return hashlib.sha256(raw).hexdigest()
def enc(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()


def build(version='ghost_hardware_v1',root=ROOT):
    if not re.fullmatch(r'ghost_hardware_v[1-9][0-9]*',version):raise ValueError('Invalid hardware probe version')
    raw=(root/'scripts/ghost_hardware_probe.py').read_bytes();source=raw.decode()
    code=('from pathlib import Path\nimport sys\nprobe_source='+repr(source)+'\n'
        'bundle=Path("/kaggle/working/ghost_hardware_bundle")\nbundle.mkdir(exist_ok=True)\n'
        '(bundle/"ghost_hardware_probe.py").write_text(probe_source)\nsys.path.insert(0,str(bundle))\n'
        'from ghost_hardware_probe import probe\nprobe("/kaggle/working/ghost_hardware_results/result.json")\n')
    ast.parse(code)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},
        'cells':[{'cell_type':'markdown','id':'scope','metadata':{},'source':['# Hardware inventory only\nRead GPU/RAM/disk inventory on the existing private free notebook. No model weights, inference, training, adapters or external service calls.\n']},
                 {'cell_type':'code','id':'probe','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    metadata=json.loads((root/'training/kaggle_cli/grounded_chat_v1/kernel-metadata.json').read_text())
    metadata.update(id_no=140792344,code_file='kaggle_ghost_hardware.ipynb',is_private=True,enable_gpu=True,enable_internet=False)
    notebook_raw=enc(notebook);metadata_raw=enc(metadata)
    lock={'notebook_sha256':sha(notebook_raw),'metadata_sha256':sha(metadata_raw),'probe_source_sha256':sha(raw),
          'purpose':'hardware_inventory_only','model_calls':0,'model_weights_loaded':False}
    stage=root/'training/kaggle_cli'/version;record=root/'evaluation'/version
    files={stage/'kaggle_ghost_hardware.ipynb':notebook_raw,stage/'kernel-metadata.json':metadata_raw,record/'bundle_lock.json':enc(lock)}
    for path,value in files.items():
        if path.exists() and path.read_bytes()!=value:raise ValueError('Frozen hardware probe changed')
    for path,value in files.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    return stage


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_hardware_v2')
    print(build(parser.parse_args().version))
