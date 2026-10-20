"""Conservative, evidence-linked extraction for explicitly labelled HLD entities and flows.

This is a document pattern extractor, not a trained AUTOSAR parser.
Only explicit declarations and 'X sends Y to Z' sentences are recognized.
"""
from __future__ import annotations
import re
import pandas as pd

ENTITY_COLUMNS = ['Type', 'Name', 'PDF page', 'Evidence excerpt', 'Review status']
REL_COLUMNS = ['Source', 'Relationship', 'Signal', 'Target', 'PDF page', 'Evidence excerpt', 'Review status']

LABELS = {
    'Software component': r'(?:Component|Software\s+Component|SWC|SW-C)',
    'Interface': r'(?:Interface|Sender[-/]Receiver\s+Interface|Client[-/]Server\s+Interface)',
    'Port': r'(?:Port|P-Port|R-Port)',
    'Signal': r'(?:Signal|Data\s+Element)',
}
IDENT = r'[A-Za-z_][A-Za-z0-9_.-]{2,79}'
# Match only complete, explicitly labelled lines. Prevent spurious matches such as
# 'Interface: completely ECU independent' or prose within PDF extracted text.
DECLARATIONS = {
    kind: re.compile(r'^\s*' + label + r'\s*[:=]\s*(' + IDENT + r')\s*[.;]?\s*$', re.I)
    for kind, label in LABELS.items()
}
FLOW = re.compile(
    r'\b(' + IDENT + r')\s+(sends|transmits|provides)\s+(' + IDENT + r')\s+to\s+(' + IDENT + r')\b',
    re.I,
)


def extract_generic_architecture(pages: list[dict]):
    entities = {}
    relationships = {}
    for page in pages:
        page_number = int(page['page'])
        for line in page.get('text', '').splitlines():
            line = ' '.join(line.split())
            if not line:
                continue
            for kind, pattern in DECLARATIONS.items():
                match = pattern.fullmatch(line)
                if match:
                    name = match.group(1)
                    key = (kind, name.casefold(), page_number)
                    entities[key] = {
                        'Type': kind, 'Name': name, 'PDF page': page_number,
                        'Evidence excerpt': line[:300], 'Review status': 'Unreviewed',
                    }
            for match in FLOW.finditer(line):
                source, verb, signal, target = match.groups()
                key = (source.casefold(), verb.casefold(), signal.casefold(), target.casefold(), page_number)
                relationships[key] = {
                    'Source': source, 'Relationship': verb.lower(), 'Signal': signal,
                    'Target': target, 'PDF page': page_number,
                    'Evidence excerpt': line[:300], 'Review status': 'Unreviewed',
                }
    return (
        pd.DataFrame(list(entities.values()), columns=ENTITY_COLUMNS),
        pd.DataFrame(list(relationships.values()), columns=REL_COLUMNS),
    )


def compare_generic_architecture(old_pages: list[dict], new_pages: list[dict]):
    old_entities, old_links = extract_generic_architecture(old_pages)
    new_entities, new_links = extract_generic_architecture(new_pages)
    def entity_set(df):
        return {(r['Type'], r['Name']) for _, r in df.iterrows()}
    def link_set(df):
        return {(r['Source'], r['Relationship'], r['Signal'], r['Target']) for _, r in df.iterrows()}
    old_e, new_e = entity_set(old_entities), entity_set(new_entities)
    old_l, new_l = link_set(old_links), link_set(new_links)
    entity_diff = [{'Change': change, 'Type': kind, 'Name': name}
                   for change, diff in [('Added', new_e-old_e), ('Removed', old_e-new_e)]
                   for kind, name in sorted(diff)]
    relation_diff = [{'Change': change, 'Source': a, 'Relationship': b, 'Signal': c, 'Target': d}
                     for change, diff in [('Added', new_l-old_l), ('Removed', old_l-new_l)]
                     for a,b,c,d in sorted(diff)]
    return (pd.DataFrame(entity_diff, columns=['Change','Type','Name']),
            pd.DataFrame(relation_diff, columns=['Change','Source','Relationship','Signal','Target']))
