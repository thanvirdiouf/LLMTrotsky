#!/usr/bin/env python3
"""Repackage the frozen v2 bundle without changing any decoded payload bytes.

The original zlib/base64 notebook exceeds SaveKernel's 1 MB source limit.
Preserve the original source, metadata and error before this transport-only change.
"""
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]


def compact(original):
    notebook = json.loads(original)
    cell = next(c for c in notebook['cells'] if c['id'] == 'bundle')
    source = ''.join(cell['source'])
    tree = ast.parse(source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'BUNDLED_FILES'
                              for target in node.targets))
    bundle = ast.literal_eval(assignment.value)
    packed, identities = {}, {}
    for name, item in bundle.items():
        raw = zlib.decompress(base64.b64decode(item['zlib_base64']))
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Original bundle hash mismatch: ' + name)
        encoded = base64.b85encode(lzma.compress(raw, preset=9)).decode('ascii')
        if lzma.decompress(base64.b85decode(encoded)) != raw:
            raise ValueError('Compact bundle roundtrip failed: ' + name)
        packed[name] = {'sha256': digest, 'lzma_base85': encoded}
        identities[name] = {'sha256': digest, 'bytes': len(raw)}
    lines = source.splitlines(keepends=True)
    lines[assignment.lineno - 1:assignment.end_lineno] = ['BUNDLED_FILES = ' + repr(packed) + '\n']
    source = ''.join(lines)
    source = source.replace('import base64,hashlib,json,sys,zlib',
                            'import base64,hashlib,json,sys,lzma')
    old = "zlib.decompress(base64.b64decode(item['zlib_base64']))"
    if source.count(old) != 1:
        raise ValueError('Unexpected frozen extraction code')
    source = source.replace(old, "lzma.decompress(base64.b85decode(item['lzma_base85']))")
    compile(source, 'compact-bundle', 'exec')
    cell['source'] = source.splitlines(keepends=True)
    for before, after in zip(json.loads(original)['cells'], notebook['cells'], strict=True):
        if before['id'] != 'bundle' and before != after:
            raise ValueError('Non-bundle cell changed')
    cli_notebook = json.loads(json.dumps(notebook))
    for c in cli_notebook['cells']:
        if c['cell_type'] == 'code':
            c['outputs'] = []
        c['source'] = ''.join(c['source'])
    normalized_size = len(json.dumps(cli_notebook).encode())
    if normalized_size >= 1_000_000:
        raise ValueError(f'Compact notebook still exceeds conservative limit: {normalized_size}')
    output = (json.dumps(notebook, ensure_ascii=False, indent=1) + '\n').encode()
    return output, {'decoded_payloads': identities, 'cli_normalized_bytes': normalized_size,
                    'original_notebook_sha256': hashlib.sha256(original).hexdigest(),
                    'compact_notebook_sha256': hashlib.sha256(output).hexdigest(),
                    'change': 'zlib/base64 to LZMA/base85 transport encoding only',
                    'all_decoded_payloads_byte_identical': True,
                    'other_cells_unchanged': True}


def main():
    stage = ROOT / 'training/kaggle_cli/data_revision_v2'
    notebook = stage / 'kaggle_data_revision_v2.ipynb'
    original = notebook.read_bytes()
    if hashlib.sha256(original).hexdigest() != 'f47e9feb665f8faf8962aa7c797b9a2538d67a03cec09ae12c340f803bd7b3cc':
        raise ValueError('Expected original frozen notebook; refusing another packaging mutation')
    output, report = compact(original)
    archive = stage / 'attempt_001_oversize'
    archive.mkdir(exist_ok=False)
    for name in ('kaggle_data_revision_v2.ipynb', 'kernel-metadata.json',
                 'submission.json', 'submission_error_body.json'):
        (archive / name).write_bytes((stage / name).read_bytes())
    (stage / 'compact_packaging_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    notebook.write_bytes(output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
