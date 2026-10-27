"""Offline compact presentation of source selections; never calls a model."""
import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path

from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError
from reviewed_source_display import records_for_packets, render_selection


def compact_case(row, records, index):
    """Revalidate the original display before presenting identical statements."""
    if row['status'] not in ('reviewed_summaries_selected', 'no_summaries_selected'):
        raise EvidenceError('Preview supports completed selections only, not blocked replies')
    original = render_selection(row['selected_claim_ids'], records)
    for key, value in original.items():
        if row.get(key) != value:
            raise EvidenceError('Original selection/display differs: ' + key)
    selected = original['source_records']
    sources = {}
    for record in selected:
        source = record['source']
        sources.setdefault(source['packet_id'], source)
    numbers = {pid: i + 1 for i, pid in enumerate(sources)}
    parts = ['<section>', '<h2>' + escape(row['case_id']) + '</h2>',
             '<p class="question">' + escape(row['question']) + '</p>']
    if not selected:
        parts.append('<p>' + escape(original['response']) + '</p>')
    else:
        parts.append('<ol class="statements">')
        for i, record in enumerate(selected):
            pid = record['source']['packet_id']
            target = f'case-{index}-record-{i}'
            kind = record.get('review_kind', 'source_summary').replace('_', ' ')
            parts.append('<li><p class="statement">' + escape(record['statement']) + '</p>'
                + '<p class="qualifier">' + escape(record['status'].replace('_', ' '))
                + ' · ' + escape(kind) + ' · <a href="#' + target + '">source '
                + str(numbers[pid]) + ', provenance</a></p>'
                + '<p class="attribution">' + escape(record['attribution']) + '</p></li>')
        parts.append('</ol><h3>Sources</h3><ol class="sources">')
        for source in sources.values():
            date = source['composition_date_interval']
            when = date['earliest']
            if when != date['latest']:
                when += ' to ' + date['latest']
            observed = '; observed ' + source['observed_date'] if 'observed_date' in source else ''
            parts.append('<li><a href="' + escape(source['url'], quote=True) + '">'
                + escape(source['title']) + '</a> — ' + escape(source['author'])
                + '; source date ' + escape(when + observed) + '</li>')
        parts.append('</ol><details><summary>Full provenance and review metadata</summary>')
        for i, record in enumerate(selected):
            parts.append(f'<article id="case-{index}-record-{i}"><h4>' + escape(record['id'])
                + '</h4><pre>' + escape(json.dumps(record, ensure_ascii=False, indent=2)) + '</pre></article>')
        parts.append('</details>')
    parts.append('</section>')
    return '\n'.join(parts), {
        'case_id': row['case_id'], 'selected_claim_ids': original['selected_claim_ids'],
        'statement_fingerprints': {r['id']: r['statement_fingerprint'] for r in selected},
        'source_record_fingerprints': original['source_record_fingerprints'],
        'original_response_fingerprint': original['response_fingerprint'],
        'all_selected_statements_preserved': True, 'all_record_metadata_preserved': True,
        'generated_factual_text_used': False, 'selection_changed': False,
        'readability_validated': False, 'requested_task_coverage_verified': False,
    }


def build_preview(bundle, answers, output):
    bundle, answers, output = Path(bundle), Path(answers), Path(output)
    collection = json.loads((bundle / 'collection.json').read_text())
    ledger = json.loads((bundle / 'ledger.json').read_text())
    jobs = json.loads((bundle / 'audit_inputs.json').read_text())
    raw = answers.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    if not rows or len({r['case_id'] for r in rows}) != len(rows):
        raise EvidenceError('Require distinct nonempty cases')
    if [r['case_id'] for r in rows] != [j['case_id'] for j in jobs]:
        raise EvidenceError('Cases differ from the original requests')
    fragments, cases = [], []
    for i, (row, job) in enumerate(zip(rows, jobs)):
        if (row['question'] != job['question']
                or row['source_context_packet_ids'] != job['source_packet_ids']):
            raise EvidenceError('Question or source scope differs from the original request')
        records = records_for_packets(collection, ledger, row['source_context_packet_ids'])
        if {cid: r['record_fingerprint'] for cid, r in records.items()} != job['available_record_fingerprints']:
            raise EvidenceError('Reviewed records differ from the original request')
        html, report = compact_case(row, records, i)
        fragments.append(html)
        cases.append(report)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Reviewed source selections — offline presentation trial</title>
<style>body{font:17px/1.55 system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 24px;color:#20252b;background:#faf9f6}section{margin:3rem 0;border-top:1px solid #bbb}h1,h2,h3{line-height:1.2}.question{background:#eeede6;padding:16px}.statement{margin-bottom:.2em}.qualifier,.attribution,.sources{font-size:.88em}.qualifier{color:#465363}.attribution{margin-top:.2em}.statements li{margin-bottom:1.3em}a{color:#215b89}details{border:1px solid #ccc;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer}article{scroll-margin-top:24px}</style>
<h1>Reviewed source selections</h1>
<p>Offline presentation trial using the original selections. These are curated
source summaries, not verbatim quotations or newly generated answers. Their
wording and attribution are unchanged. Record status describes the source
claim, not independent verification. Missing selections and incomplete answers
remain unresolved. Full provenance includes speakers, roles, source boundaries,
anchors and application-derived inferences.</p>
''' + '\n'.join(fragments) + '</html>\n'
    report = {'purpose': 'offline_presentation_replay', 'model_calls': 0,
              'answers_sha256': sha256(raw).hexdigest(),
              'requests_sha256': sha256((bundle / 'audit_inputs.json').read_bytes()).hexdigest(),
              'preview_script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
              'answers_file_fingerprint': fingerprint(raw.decode()),
              'collection_packets_fingerprint': collection['packets_sha256'],
              'ledger_fingerprint': fingerprint(ledger), 'cases': cases,
              'preview_fingerprint': fingerprint(page), 'active_route_changed': False,
              'source_quality_independently_validated': False, 'readability_validated': False}
    # Validate the whole replay before creating output; never replace a prior trial.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'preview.html').write_text(page)
    (output / 'replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--answers', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = build_preview(args.bundle, args.answers, args.output)
    print(json.dumps({'cases': len(report['cases']), 'model_calls': 0,
                      'readability_validated': False, 'output': args.output}))


if __name__ == '__main__':
    main()
