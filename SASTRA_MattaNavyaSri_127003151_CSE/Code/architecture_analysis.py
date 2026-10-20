"""Conservative, evidence-linked AUTOSAR Basic Software inventory and relationships.

This is a rule-based pilot for the supplied layered-architecture specification,
not a general-purpose extractor of arbitrary software-component design documents.
"""
import re
import pandas as pd

# Explicitly classify Basic Software modules rather than calling every module an SW-C.
MODULE_CATALOG = {
    "AUTOSAR Runtime Environment (RTE)": "Runtime environment",
    "NVRAM Manager": "Basic Software module",
    "Memory Abstraction Interface": "Basic Software interface",
    "SPIHandlerDriver": "Basic Software driver",
    "Watchdog Manager": "Basic Software module",
    "PDU Router": "Basic Software module",
    "AUTOSAR COM": "Basic Software module",
    "Diagnostic Communication Manager": "Basic Software module",
    "CAN Interface": "Basic Software interface",
    "LIN Interface": "Basic Software interface",
    "FlexRay Interface": "Basic Software interface",
    "ECU State Manager": "Basic Software module",
    "Communication Manager": "Basic Software module",
    "CAN Driver": "Basic Software driver",
    "LIN Driver": "Basic Software driver",
}

# Relationships are added ONLY when a page explicitly contains these statements.
# Conservative patterns include flexible whitespace for PDF extraction artifacts.
RULES = [
    ("NVRAM Manager", "accesses drivers through", "Memory Abstraction Interface",
     r"NVRAM\s+Manager\s+accesses\s+drivers\s+via\s+the\s+Memory\s+Abstraction.{0,55}?Interface"),
    ("SPIHandlerDriver", "controls access to", "SPI hardware",
     r"SPIHandlerDriver\s+controls\s+the\s+concurrent\s+access\s+to\s+the\s+SPI.{0,85}?hardware"),
    ("AUTOSAR software components", "communicate through", "AUTOSAR Runtime Environment (RTE)",
     r"AUTOSAR\s+Software\s+Components\s+communicate\s+with\s+other\s+components.{0,110}?via\s+the\s+RTE"),
    ("PDU Router", "routes PDUs between", "communication controllers and upper layers",
     r"Provides\s+routing\s+of\s+PDUs\s+between\s+different\s+abstract\s+communication.{0,60}?controllers\s+and\s+upper\s+layers"),
]


def analyze_architecture(pages: list[dict]):
    entity_rows = []
    relationship_rows = []
    for name, kind in MODULE_CATALOG.items():
        matches = [p['page'] for p in pages if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', p['text'], re.I)]
        if matches:
            entity_rows.append({'Name': name, 'Classification': kind,
                                'PDF pages': ', '.join(map(str, matches[:15])),
                                'Mentions on pages': len(matches), 'Review status': 'Unreviewed'})
    for page in pages:
        text = re.sub(r'\s+', ' ', page['text'])
        for source, relation, target, pattern in RULES:
            for match in re.finditer(pattern, text, re.I):
                start, end = match.span()
                evidence = text[max(0, start - 75):min(len(text), end + 100)].strip()
                relationship_rows.append({'Source': source, 'Relationship': relation,
                                          'Target': target, 'PDF page': page['page'],
                                          'Evidence excerpt': evidence,
                                          'Review status': 'Unreviewed'})
    entities = pd.DataFrame(entity_rows, columns=['Name','Classification','PDF pages','Mentions on pages','Review status'])
    relationships = pd.DataFrame(relationship_rows, columns=['Source','Relationship','Target','PDF page','Evidence excerpt','Review status'])
    return entities, relationships
