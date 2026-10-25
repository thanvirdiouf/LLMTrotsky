"""Render descriptive tables only after scores have been locked and mapped."""
import json
from pathlib import Path
from review_grounding import RUN, CONDITIONS

def write_report():
    q=json.loads((RUN/'quality_summary.json').read_bytes())
    labels=dict(zip(CONDITIONS,['Base, no passages','Base, passages','Saved 1e-4, no passages','Saved 1e-4, passages']))
    cats=['chronology','false_premise','invented_memory_citation','answer_detail']
    lines=['# Source-grounding results — 25 October 2026','',
      'All 96 requested answers were generated on Kaggle without training. The 24 fresh cases, source passages, prompts and scoring rules were frozen before generation. A separate coding agent with clean task context reviewed randomized neutral answers; scores and answer hashes were locked before the condition key was mapped. This is not independent historian validation, and answer style may reveal conditions.','',
      'Each critical assertion forces the primary score to zero. Separate dimension scores and fulfilled source-check indices remain in the audit. Means below describe six ordinal judgments per category; they are not a pooled quality metric.','',
      '| Condition | Chronology | Premise | Evidence calibration | Useful explanation | Critical cases |',
      '|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        s=q['conditions'][c];lines.append('| '+labels[c]+' | '+' | '.join(f"{s['category_means'][k]:.2f}" for k in cats)+f" | {s['critical_cases']}/24 |")
    lines+=['','## Paired comparisons','','Improved/degraded/tied counts use the same case’s critical-zero primary score; the categories remain separate in the JSON and CSV.','','| Comparison | Improved | Degraded | Tied |','|---|---:|---:|---:|']
    for name,s in q['paired'].items():lines.append(f"| {name} | {s['improved']} | {s['degraded']} | {s['tied']} |")
    lines+=['','## Premise handling, fabrication and supported answers','','| Condition | Corrected negative premises | Fabricated scenes/quotes/citations | Unsupported replacement explanations | Supported positive answers | Positive score ≥3 | Blanket refusals | Detail score ≥3 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        s=q['conditions'][c];lines.append(f"| {labels[c]} | {s['premises_corrected']}/{s['negative_premise_n']} | {s['fabrication_cases']} | {s['unsupported_replacements']} | {s['positive_supported']}/{s['positive_n']} | {s['positive_score_ge3']}/{s['positive_n']} | {s['positive_refusals']}/{s['positive_n']} | {s['detail_score_ge3']}/6 |")
    lines+=['','The supported-answer flag and score threshold are both reported to expose borderline judgments. Full responses, exact reviewer notes and all source checks are in `CASE_AUDIT.md` and `review/`; do not infer quality from length alone.','','## Source clustering','','Four related cases per selected work; the known near-duplicate family was reserved too. Each cell shows chronology/premise/evidence/detail primary scores for that source.','','| Source family | Base, no passages | Base, passages | 1e-4, no passages | 1e-4, passages |','|---|---|---|---|---|']
    source_names={c['source_family_id']:c['case_id'].split('_')[1] for c in [json.loads(s) for s in (RUN/'review/cases.jsonl').read_text().splitlines()]}
    for fam,groups in q['source_level'].items():lines.append('| '+source_names[fam]+' ('+fam+') | '+' | '.join('/'.join(str(groups[c]['category_means'][k]) for k in cats) for c in CONDITIONS)+' |')
    lines+=['','## Length diagnostics','','| Condition | Mean output tokens | Reached 512-token cap |','|---|---:|---:|']
    for c in CONDITIONS:
        s=q['length_diagnostics'][c];lines.append(f"| {labels[c]} | {s['mean_tokens']} | {s['truncations']}/24 |")
    lines+=['','## Remaining failures with clear passages','','These answers require inspection before any proposal for targeted training. “Below3” includes incomplete answers as well as mistakes; it does not by itself mean fabrication.','']
    mapped=[json.loads(s) for s in (RUN/'review/mapped_scores.jsonl').read_text().splitlines()]
    for c in [CONDITIONS[1],CONDITIONS[3]]:
        failures=[s for s in mapped if s['condition']==c and (s['critical_zero_score']<3 or s['critical_types'])]
        lines+=['### '+labels[c],'']
        if not failures:lines+=['None under the frozen rubric.','']
        for s in failures:lines += [f"- **{s['case_id']}** — primary {s['critical_zero_score']}, dimension {s['dimension_score']}; critical {s['critical_types'] or 'none'}. {s['notes']}"]
        lines+=['']
    lines+=['## Interpretation limits','',*['- '+s for s in q['limitations']],'- These cases are now inspected development evidence; future primary tests need new families.','- Both adapters remain experimental. No extra epoch, learning-rate sweep or model promotion occurred.','',
      'The intervention decision is recorded separately in `DECISION.md` after inspecting the paired results and residual grounded failures. All original/revised training data, saved adapters and earlier locked reviews remain preserved.','']
    (RUN/'QUALITY_REVIEW.md').write_text('\n'.join(lines))

if __name__=='__main__':write_report()
