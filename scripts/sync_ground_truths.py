"""Sync data/ground_truths.json from the airline_carrier ground-truth YAML.

Maps the YAML schema (id/question/reference_answer) to this repo's schema and
attaches per-question metadata: a descriptive category and the list of manual
sections each expected answer draws on. Source sections are validated against
the generated corpus JSON, so a drifting section number fails loudly here
rather than silently breaking the retrieval check.

Usage:
    python scripts/sync_ground_truths.py

Re-run whenever the YAML changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_SOURCE = Path(
    "/home/anupamkrishnamurthy/repos/ai-eval-platform/demos/domains/"
    "airline_carrier/ground_truth.yml"
)
CORPUS_PATH = REPO_ROOT / "data" / "skyway" / "customer-service-reference-manual.json"
DST_PATH = REPO_ROOT / "data" / "ground_truths.json"
HTML_URL = "data/skyway/customer-service-reference-manual.html"

# Per-question category and the manual sections the expected answer draws on
# (derived from the YAML `notes`). Order: primary section first.
QUESTION_META = {
    "AQ0001": ("multi-step-deduction", ["4.5", "5.2", "5.3", "6.3"]),
    "AQ0002": ("comparative-policy", ["4.3", "4.2"]),
    "AQ0003": ("quantitative-fact", ["3.2"]),
    "AQ0004": ("conditional-policy", ["4.2", "4.1"]),
    "AQ0005": ("multi-part-policy", ["5.2", "5.3"]),
    "AQ0006": ("comparative-policy", ["2.4"]),
    "AQ0007": ("multi-part-list", ["2.1"]),
    "AQ0008": ("multi-part-list", ["6.3"]),
    "AQ0009": ("negative-test", ["7.5"]),
    "AQ0010": ("debatable-policy", ["3.6"]),
    "AQ0011": ("multi-part-policy", ["5.4", "5.2"]),
    "AQ0012": ("conditional-policy", ["4.2"]),
    "AQ0013": ("multi-part-policy", ["3.6"]),
    "AQ0014": ("comparative-policy", ["4.1"]),
    "AQ0015": ("conditional-policy", ["3.1", "3.2"]),
    "AQ0016": ("conditional-policy", ["6.5"]),
}


def load_section_index() -> dict:
    """section id -> {"title": ..., "url": ...} from the corpus JSON."""
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    index = {}
    for entry in corpus:
        for section in entry["sections"]:
            anchor = "section-" + section["id"].replace(".", "-")
            index[section["id"]] = {
                "title": section["title"],
                "url": f"{HTML_URL}#{anchor}",
            }
    return index


def main() -> None:
    sections = load_section_index()

    with open(YAML_SOURCE, "r", encoding="utf-8") as f:
        items = yaml.safe_load(f)["ground_truth"]

    yaml_ids = {item["id"] for item in items}
    meta_ids = set(QUESTION_META)
    if yaml_ids != meta_ids:
        sys.exit(
            f"Question set drift — update QUESTION_META in {__file__}.\n"
            f"  only in YAML: {sorted(yaml_ids - meta_ids)}\n"
            f"  only in QUESTION_META: {sorted(meta_ids - yaml_ids)}"
        )

    # evaluation_criteria is hand-maintained in ground_truths.json; preserve it.
    existing_criteria: dict = {}
    if DST_PATH.exists():
        with open(DST_PATH, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                if "evaluation_criteria" in entry:
                    existing_criteria[entry["id"]] = entry["evaluation_criteria"]

    out = []
    for item in items:
        category, section_ids = QUESTION_META[item["id"]]
        unknown = [s for s in section_ids if s not in sections]
        if unknown:
            sys.exit(f"{item['id']}: unknown section ids {unknown} — not in corpus.")
        out.append(
            {
                "id": item["id"],
                "category": category,
                "question": item["question"].strip(),
                "evaluation_criteria": existing_criteria.get(item["id"], []),
                "sources": [
                    {
                        "id": sid,
                        "title": sections[sid]["title"],
                        "url": sections[sid]["url"],
                    }
                    for sid in section_ids
                ],
            }
        )

    with open(DST_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(out)} entries to {DST_PATH}")
    for e in out:
        ids = ", ".join(s["id"] for s in e["sources"])
        print(f"  {e['id']} ({e['category']}) sources: {ids}")


if __name__ == "__main__":
    main()
