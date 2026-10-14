"""Accuracy numbers for the Results page, read from the committed reports in results/."""

from __future__ import annotations

import re
from pathlib import Path

from app.settings import ROOT

RESULTS = ROOT / "results"


def _cell(text: str) -> dict:
    t = text.strip()
    bold = t.startswith("**")
    t = t.replace("**", "").replace("`", "")
    return {"text": t, "bold": bold}


def markdown_tables(path: Path) -> dict[str, list[dict]]:
    """Tables per section heading: {heading: [{"header": [...], "rows": [[cell, ...], ...]}]}."""
    out: dict[str, list[dict]] = {}
    heading, table = "", None
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            heading, table = line.lstrip("# ").strip(), None
            continue
        if line.startswith("|"):
            cells = [_cell(c) for c in line.strip().strip("|").split("|")]
            if table is None:
                table = {"header": [c["text"] for c in cells], "rows": []}
                out.setdefault(heading, []).append(table)
            elif not all(re.fullmatch(r":?-+:?", c["text"]) for c in cells):
                table["rows"].append(cells)
        else:
            table = None
    return out


LABELS = {"vin": "VIN", "plate_state": "Plate state", "invoice_date": "Invoice date", "invoice_no": "Invoice no."}
_PAIR = re.compile(r"(\d+(?:\.\d+)?)%? -> (\d+(?:\.\d+)?)%?")


def invoice_fields() -> dict | None:
    """Per-field accuracy, rules-only vs rules + vision model, for pdf / scan / photo."""
    path = RESULTS / "rules_vs_ai_hard_dev.md"
    if not path.exists():
        return None
    block = re.search(r"```\n(.*?)```", path.read_text(), re.S)
    if not block:
        return None
    fields, totals, counts = [], None, {}
    for line in block.group(1).splitlines():
        m = re.match(r"^([A-Za-z_+ ]+?)\s{2,}(.*)$", line.strip())
        if not m:
            continue
        name, rest = m.group(1).strip(), m.group(2)
        pairs = [(float(a), float(b)) for a, b in _PAIR.findall(rest)]
        if len(pairs) != 3:
            continue
        cols = dict(zip(("pdf", "scan", "photo"), pairs))
        if "%" in rest and name == "all fields":
            totals = cols
        elif "%" in rest:
            fields.append({"name": name, "label": LABELS.get(name, name.replace("_", " ").capitalize()), **cols})
        else:
            counts[name] = cols
    return {"fields": fields, "totals": totals, "counts": counts} if fields else None


def matching() -> dict | None:
    """Clusters auto-linked / review / unmatched, rules-only vs with the model."""
    path = RESULTS / "matching_dev.md"
    if not path.exists():
        return None
    tables = markdown_tables(path)
    main = next(iter(tables.get("Results", [])), None)
    if main is None:
        return None
    out = {"table": main, "hard_cases": next(iter(tables.get("Hard cases", [])), None), "bars": {}}
    for row in main["rows"]:
        label = row[0]["text"]
        key = {"Clusters auto-linked": "auto", "Clusters sent to review": "review",
               "Clusters unmatched": "unmatched", "Wrong auto-links (target 0)": "wrong"}.get(label)
        if key:
            for i, mode in enumerate(("rules", "ai"), start=1):
                m = re.search(r"(\d+(?:\.\d+)?)%", row[i]["text"]) if key != "wrong" else None
                n = re.match(r"(\d+)", row[i]["text"])
                out["bars"].setdefault(mode, {})[key] = float(m.group(1)) if m else int(n.group(1)) if n else 0
    misread = [t for h, ts in tables.items() if h.startswith("Likely misread") for t in ts]
    out["misread"] = misread[0] if misread else None
    return out


def photos() -> dict | None:
    path = RESULTS / "photo_accuracy_dev.md"
    if not path.exists():
        return None
    tables = markdown_tables(path)
    per_type = next(iter(tables.get("Per photo type", [])), None)
    trust = next(iter(tables.get("Trust (target: 0 wrong + trusted)", [])), None)
    return {"per_type": per_type, "trust": trust} if per_type or trust else None


def load() -> dict:
    return {"invoice": invoice_fields(), "matching": matching(), "photos": photos()}
