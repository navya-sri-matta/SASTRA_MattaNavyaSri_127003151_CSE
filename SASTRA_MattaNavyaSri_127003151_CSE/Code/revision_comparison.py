"""Local, deterministic PDF revision comparison for AUTOSAR architecture documents.

Text comparison is advisory: OCR/formatting changes, page shifts and diagrams may
produce false differences or hide real architectural changes. No LLM required.
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

from architecture_analysis import analyze_architecture

COLUMNS = ["Change", "Old PDF page", "New PDF page", "Old excerpt", "New excerpt", "Review status"]


def _clean_lines(pages: list[dict]) -> list[tuple[str, int]]:
    """Normalize whitespace but keep page numbers for traceability."""
    result: list[tuple[str, int]] = []
    for page in pages:
        for line in page["text"].splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) < 4:
                continue
            # Avoid comparing repeated document running headers and footers.
            if re.fullmatch(r"(?:Page\s+\d+|[-–]?\s*AUTOSAR Confidential\s*[-–]?)", line, re.I):
                continue
            result.append((line, page["page"]))
    return result


def _excerpt(lines: list[tuple[str, int]], max_characters: int = 500) -> str:
    return " | ".join(line for line, _ in lines)[:max_characters]


def compare_revisions(old_pages: list[dict], new_pages: list[dict], max_rows: int = 150):
    """Return (text changes, inventory changes, relation changes, statistics)."""
    old_lines = _clean_lines(old_pages)
    new_lines = _clean_lines(new_pages)
    before = [line for line, _ in old_lines]
    after = [line for line, _ in new_lines]
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    rows = []
    total_changes = 0
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation == "equal":
            continue
        total_changes += 1
        if len(rows) >= max_rows:
            continue
        left = old_lines[i1:i2]
        right = new_lines[j1:j2]
        rows.append({
            "Change": {"replace": "Modified", "delete": "Removed", "insert": "Added"}[operation],
            "Old PDF page": left[0][1] if left else None,
            "New PDF page": right[0][1] if right else None,
            "Old excerpt": _excerpt(left),
            "New excerpt": _excerpt(right),
            "Review status": "Unreviewed",
        })
    changes = pd.DataFrame(rows, columns=COLUMNS)

    old_inventory, old_relations = analyze_architecture(old_pages)
    new_inventory, new_relations = analyze_architecture(new_pages)
    old_names = set(old_inventory["Name"])
    new_names = set(new_inventory["Name"])
    inventory_rows = ([{"Change": "Added", "Element": n} for n in sorted(new_names - old_names)]
                      + [{"Change": "Removed", "Element": n} for n in sorted(old_names - new_names)])
    inventory_changes = pd.DataFrame(inventory_rows, columns=["Change", "Element"])

    def relations_key(frame):
        return {(str(r["Source"]), str(r["Relationship"]), str(r["Target"])) for _, r in frame.iterrows()}

    old_links = relations_key(old_relations)
    new_links = relations_key(new_relations)
    relation_rows = ([{"Change": "Added", "Source": a, "Relationship": b, "Target": c}
                      for a, b, c in sorted(new_links - old_links)]
                     + [{"Change": "Removed", "Source": a, "Relationship": b, "Target": c}
                        for a, b, c in sorted(old_links - new_links)])
    relation_changes = pd.DataFrame(relation_rows, columns=["Change", "Source", "Relationship", "Target"])
    stats = {"old_pages": len(old_pages), "new_pages": len(new_pages),
             "change_blocks": total_changes, "displayed_blocks": len(rows),
             "unchanged_line_ratio": matcher.ratio()}
    return changes, inventory_changes, relation_changes, stats
