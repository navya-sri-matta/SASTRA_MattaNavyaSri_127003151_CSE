"""Conservative checks over explicit HLD declarations and signal-flow sentences.

Findings are potential document inconsistencies, not proof of system defects.
"""
from __future__ import annotations
import pandas as pd
from generic_extraction import extract_generic_architecture

COLUMNS = ['Severity', 'Check', 'Subject', 'PDF page', 'Explanation', 'Evidence excerpt', 'Review status']

def analyze_inconsistencies(pages: list[dict]) -> pd.DataFrame:
    entities, flows = extract_generic_architecture(pages)
    findings = []
    def record(severity, check, subject, page, explanation, evidence):
        findings.append({'Severity': severity, 'Check': check, 'Subject': subject,
                         'PDF page': int(page), 'Explanation': explanation,
                         'Evidence excerpt': str(evidence)[:350], 'Review status': 'Unreviewed'})
    # Name matching ignores capitalization. All comparisons are document-local.
    by_name = {}
    declared_by_type = {}
    for _, row in entities.iterrows():
        name = str(row['Name'])
        norm = name.casefold()
        kind = str(row['Type'])
        by_name.setdefault(norm, []).append(row)
        declared_by_type.setdefault(kind, set()).add(norm)
    # A name declared as multiple incompatible kinds could be an authoring error.
    for norm, declarations in by_name.items():
        kinds = sorted({str(r['Type']) for r in declarations})
        if len(kinds) > 1:
            first = declarations[0]
            pages_involved = sorted({int(r['PDF page']) for r in declarations})
            record('Warning', 'Conflicting declaration types', str(first['Name']),
                   first['PDF page'], f'Name declared under multiple types: {", ".join(kinds)}; pages {pages_involved}.',
                   ' | '.join(str(r['Evidence excerpt']) for r in declarations))
    components = declared_by_type.get('Software component', set())
    signals = declared_by_type.get('Signal', set())
    for _, flow in flows.iterrows():
        page = flow['PDF page']; evidence = flow['Evidence excerpt']
        for role in ('Source', 'Target'):
            name = str(flow[role])
            if name.casefold() not in components:
                record('Warning', 'Undeclared component reference', name, page,
                       f'{role} component in a signal flow has no explicit Component declaration in this uploaded PDF.', evidence)
        signal = str(flow['Signal'])
        if signal.casefold() not in signals:
            record('Warning', 'Undeclared signal reference', signal, page,
                   'Signal used in a flow has no explicit Signal declaration in this uploaded PDF.', evidence)
    # Multiple declarations of the same type and name across pages are not automatically errors.
    # Interface/port mappings are not inferred from mere co-occurrence.
    return pd.DataFrame(findings, columns=COLUMNS)
