#!/usr/bin/env python3
"""Safely unpack the preserved lower-LR delivery and run strict local verification."""
import argparse,hashlib,json,shutil,stat,tempfile,zipfile
from pathlib import Path
from export_lr_test_review import validate_run
ROOT=Path(__file__).resolve().parents[1]
def receive(directory):
 directory=Path(directory).resolve();target=directory/'results'
 if target.exists():raise FileExistsError('Preserve existing results; refusing to overwrite')
 files=list((directory/'download').rglob('lr_test_delivery.zip'))
 if len(files)!=1:raise ValueError('Expected exactly one preserved delivery ZIP')
 archive_path=files[0];h=hashlib.sha256()
 with archive_path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 with zipfile.ZipFile(archive_path) as archive:
  entries=archive.infolist();names=[r.filename for r in entries]
  if len(names)!=len(set(names)):raise ValueError('Duplicate archive path')
  for item in entries:
   p=Path(item.filename)
   if (p.is_absolute() or '..' in p.parts or '\\' in item.filename or not item.filename
       or item.is_dir() or stat.S_ISLNK(item.external_attr>>16)):
    raise ValueError('Unsafe delivery archive path/type')
  if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
  temporary=Path(tempfile.mkdtemp(prefix='.lr-delivery-',dir=directory))
  try:
   for item in entries:
    destination=temporary/item.filename;destination.parent.mkdir(parents=True,exist_ok=True)
    with archive.open(item) as source,destination.open('xb') as output:shutil.copyfileobj(source,output)
   temporary.rename(target)
  finally:
   if temporary.exists():shutil.rmtree(temporary)
 # Strict verification reads journals internally but prints no model responses or labels.
 _,_,_,_,verification=validate_run(target,project=ROOT,archive_path=archive_path)
 verification['safe_archive_checks']={'archive':str(archive_path.relative_to(ROOT)),'archive_sha256':h.hexdigest(),'bytes':archive_path.stat().st_size,'entries':len(entries),'crc_passed':True,'safe_unique_paths':True}
 (directory/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
 print(json.dumps(verification,indent=2))
 return target
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();receive(args.directory)
