"""Build pipeline for the SkyWay Customer Service Reference Manual.

Parses the plain-text manual (the single source of truth) and regenerates:
  1. A self-contained, browser-readable HTML version.
  2. The chunked JSON corpus consumed by the RAG system (one section entry
     per top-level chapter, one section per numbered subsection).

Usage:
    python scripts/build_manual.py

Re-run whenever the TXT manual changes.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
SKYWAY_DIR = REPO_ROOT / "data" / "skyway"
TXT_PATH = SKYWAY_DIR / "customer-service-reference-manual.txt"
HTML_PATH = SKYWAY_DIR / "customer-service-reference-manual.html"
JSON_PATH = SKYWAY_DIR / "customer-service-reference-manual.json"

# Relative URL used in the corpus metadata; anchors point into the HTML file.
HTML_URL = "data/skyway/customer-service-reference-manual.html"

TOP_HEADING_RE = re.compile(r"^(\d+)\. (.+)$")
SUB_HEADING_RE = re.compile(r"^(\d+)\.(\d+)\. (.+)$")
BULLET_RE = re.compile(r"^(\s*)•\s+(.*)$")
NUM_ITEM_RE = re.compile(r"^(\d+)\.\s+(.*)$")


@dataclass
class Subsection:
    number: str  # e.g. "6.3"
    title: str  # e.g. "Tier Levels"
    lines: List[str] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return f"{self.number}. {self.title}"

    @property
    def anchor(self) -> str:
        return "section-" + self.number.replace(".", "-")

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip("\n")


@dataclass
class Section:
    number: int  # e.g. 6
    title: str  # e.g. "SkyWay Voyager Loyalty Program"
    intro_lines: List[str] = field(default_factory=list)
    subsections: List[Subsection] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return f"{self.number}. {self.title}"

    @property
    def anchor(self) -> str:
        return f"section-{self.number}"


@dataclass
class Manual:
    title_lines: List[str] = field(default_factory=list)
    preamble_lines: List[str] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)

    @property
    def preamble_text(self) -> str:
        return "\n".join(self.preamble_lines).strip("\n")


def _is_blank_or_eof(lines: List[str], index: int) -> bool:
    return index >= len(lines) or not lines[index].strip()


def parse_manual(text: str) -> Manual:
    lines = text.split("\n")
    manual = Manual()
    section: Section | None = None
    subsection: Subsection | None = None
    expected_top = 1

    # Leading block: document title lines up to the first blank line,
    # then preamble paragraphs until the first top-level heading.
    i = 0
    while i < len(lines) and lines[i].strip():
        manual.title_lines.append(lines[i].strip())
        i += 1

    for idx in range(i, len(lines)):
        line = lines[idx].rstrip()

        # Subsection heading, e.g. "6.3. Tier Levels". Must belong to the
        # current top-level section and be followed by a blank line, which
        # rules out numbered list items in the body text.
        m_sub = SUB_HEADING_RE.match(line)
        if (
            m_sub
            and section is not None
            and int(m_sub.group(1)) == section.number
            and _is_blank_or_eof(lines, idx + 1)
        ):
            subsection = Subsection(
                number=f"{m_sub.group(1)}.{m_sub.group(2)}",
                title=m_sub.group(3).strip(),
            )
            section.subsections.append(subsection)
            continue

        # Top-level heading, e.g. "6. SkyWay Voyager Loyalty Program".
        # Headings appear in strict 1..N order, which rules out numbered
        # list items such as "1. The passenger must report ..." in §3.6.
        m_top = TOP_HEADING_RE.match(line)
        if (
            m_top
            and not m_sub
            and int(m_top.group(1)) == expected_top
            and _is_blank_or_eof(lines, idx + 1)
        ):
            section = Section(number=expected_top, title=m_top.group(2).strip())
            manual.sections.append(section)
            subsection = None
            expected_top += 1
            continue

        if subsection is not None:
            subsection.lines.append(line)
        elif section is not None:
            section.intro_lines.append(line)
        else:
            manual.preamble_lines.append(line)

    # Drop the closing line ("End of ... Manual.") from the last subsection.
    if manual.sections and manual.sections[-1].subsections:
        last = manual.sections[-1].subsections[-1]
        while last.lines and not last.lines[-1].strip():
            last.lines.pop()
        if last.lines and last.lines[-1].strip().lower().startswith("end of "):
            last.lines.pop()

    return manual


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def render_body(body_lines: List[str]) -> str:
    """Render a subsection body (paragraphs, bullets, numbered steps,
    and term blocks) to HTML."""
    out: List[str] = []
    para: List[str] = []
    items: List[List[str]] = []
    list_tag: str | None = None  # "ul" or "ol" while a list is open
    bullet_indent = 0

    def flush_para() -> None:
        nonlocal para
        if para:
            text = " ".join(s.strip() for s in para)
            out.append(f"<p>{html.escape(text)}</p>")
            para = []

    def flush_list() -> None:
        nonlocal items, list_tag
        if items and list_tag:
            lis = "".join(
                f"<li>{html.escape(' '.join(s.strip() for s in item))}</li>"
                for item in items
            )
            out.append(f"<{list_tag} class=\"{'steps' if list_tag == 'ol' else 'bullets'}\">{lis}</{list_tag}>")
        items = []
        list_tag = None

    for idx, raw in enumerate(body_lines):
        line = raw.rstrip()
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if not stripped:
            flush_para()
            flush_list()
            continue

        m_bullet = BULLET_RE.match(line)
        if m_bullet:
            flush_para()
            if list_tag == "ol":
                flush_list()
            list_tag = "ul"
            bullet_indent = len(m_bullet.group(1))
            items.append([m_bullet.group(2)])
            continue

        m_num = NUM_ITEM_RE.match(line)
        if m_num and indent == 0:
            flush_para()
            if list_tag == "ul":
                flush_list()
            list_tag = "ol"
            items.append([m_num.group(2)])
            continue

        # Continuation of an open list item (more deeply indented line).
        if list_tag == "ul" and indent > bullet_indent and items:
            items[-1].append(stripped)
            continue
        if list_tag == "ol" and indent > 0 and items:
            items[-1].append(stripped)
            continue

        # A column-0 line directly followed by an indented line is a term
        # heading (e.g. "Economy Class", "Sports equipment:").
        next_line = body_lines[idx + 1] if idx + 1 < len(body_lines) else ""
        next_indent = len(next_line) - len(next_line.lstrip()) if next_line.strip() else 0
        if indent == 0 and not para and next_line.strip() and next_indent > 0:
            flush_list()
            out.append(f"<h4>{html.escape(stripped)}</h4>")
            continue

        flush_list()
        para.append(stripped)

    flush_para()
    flush_list()
    return "\n".join(out)


CSS = """
:root {
  --sky-blue: #0057a8;
  --sky-dark: #003b73;
  --sky-light: #e8f0fa;
  --text: #2c2c2c;
  --text-muted: #5a5a5a;
  --border: #d0d7de;
  --bg: #fff;
  --section-bg: #f6f8fa;
  --sidebar-width: 280px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: var(--text);
  line-height: 1.7;
  background: var(--sky-light);
}
header {
  background: linear-gradient(135deg, var(--sky-dark), var(--sky-blue));
  color: #fff;
  padding: 2rem 2rem 2rem calc(var(--sidebar-width) + 2rem);
}
header h1 { font-size: 2rem; font-weight: 700; letter-spacing: 0.02em; }
header p { margin-top: 0.5rem; font-size: 1rem; opacity: 0.85; }
.layout { display: flex; min-height: calc(100vh - 6rem); }
nav {
  position: fixed;
  top: 0;
  left: 0;
  width: var(--sidebar-width);
  height: 100vh;
  overflow-y: auto;
  background: var(--bg);
  border-right: 1px solid var(--border);
  padding: 1.25rem 1rem;
  z-index: 10;
}
nav .nav-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--sky-dark);
  margin-bottom: 0.25rem;
}
nav .nav-subtitle {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
}
nav h2 {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
  border-bottom: none;
  padding-bottom: 0;
}
nav ol { list-style: none; }
nav > ol > li { margin-bottom: 0.6rem; }
nav > ol > li > a {
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--sky-dark);
}
nav ol ol { margin-top: 0.25rem; padding-left: 0.75rem; }
nav ol ol li { margin-bottom: 0.15rem; }
nav ol ol a { font-size: 0.8rem; font-weight: 400; }
nav a {
  color: var(--sky-blue);
  text-decoration: none;
  display: block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  transition: background 0.15s;
}
nav a:hover { background: var(--sky-light); text-decoration: none; }
nav a.active { background: var(--sky-blue); color: #fff; }
main {
  margin-left: var(--sidebar-width);
  flex: 1;
  background: var(--bg);
  padding: 2rem 3rem 4rem;
  max-width: 56rem;
}
.preamble {
  border-left: 4px solid var(--sky-blue);
  background: var(--sky-light);
  padding: 1rem 1.25rem;
  margin-bottom: 2.5rem;
  font-size: 0.95rem;
  color: var(--text-muted);
}
section { margin-bottom: 2.5rem; scroll-margin-top: 1.5rem; }
main h2 {
  font-size: 1.5rem;
  color: var(--sky-dark);
  border-bottom: 2px solid var(--sky-blue);
  padding-bottom: 0.4rem;
  margin-bottom: 1.25rem;
  scroll-margin-top: 1.5rem;
}
h3 {
  font-size: 1.15rem;
  color: var(--sky-blue);
  margin-top: 1.75rem;
  margin-bottom: 0.75rem;
  scroll-margin-top: 1.5rem;
}
h4 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--sky-dark);
  margin-top: 1.25rem;
  margin-bottom: 0.5rem;
}
p { margin-bottom: 0.75rem; }
ul.bullets, ol.steps { margin-left: 1.5rem; margin-bottom: 0.75rem; }
li { margin-bottom: 0.35rem; }
ol.steps { counter-reset: step; list-style: none; margin-left: 0; }
ol.steps li {
  counter-increment: step;
  padding-left: 2rem;
  position: relative;
  margin-bottom: 0.6rem;
}
ol.steps li::before {
  content: counter(step);
  position: absolute;
  left: 0;
  width: 1.4rem;
  height: 1.4rem;
  border-radius: 50%;
  background: var(--sky-blue);
  color: #fff;
  text-align: center;
  font-size: 0.8rem;
  line-height: 1.4rem;
  font-weight: 600;
}
footer {
  text-align: center;
  padding: 2rem;
  font-size: 0.85rem;
  color: var(--text-muted);
  background: var(--bg);
  margin-left: var(--sidebar-width);
  max-width: 56rem;
  border-top: 1px solid var(--border);
}
@media (max-width: 768px) {
  :root { --sidebar-width: 0px; }
  nav { display: none; }
  header { padding: 2rem; }
  main { padding: 1.25rem; margin-left: 0; }
  footer { margin-left: 0; }
}
"""

TOC_JS = """
document.addEventListener('DOMContentLoaded', function() {
  var links = document.querySelectorAll('nav a');
  var headings = [];
  links.forEach(function(a) {
    var id = a.getAttribute('href').substring(1);
    var el = document.getElementById(id);
    if (el) headings.push({el: el, link: a});
  });
  function updateActive() {
    var scrollY = window.scrollY + 80;
    var current = null;
    for (var i = 0; i < headings.length; i++) {
      if (headings[i].el.offsetTop <= scrollY) current = headings[i];
    }
    links.forEach(function(a) { a.classList.remove('active'); });
    if (current) current.link.classList.add('active');
  }
  window.addEventListener('scroll', updateActive, {passive: true});
  updateActive();
  links.forEach(function(a) {
    a.addEventListener('click', function(e) {
      var id = this.getAttribute('href').substring(1);
      var target = document.getElementById(id);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({behavior: 'smooth', block: 'start'});
        history.replaceState(null, '', '#' + id);
      }
    });
  });
});
"""


def render_html(manual: Manual) -> str:
    doc_title = manual.title_lines[0] if manual.title_lines else "Reference Manual"
    subtitle = " — ".join(manual.title_lines[1:])

    toc_items: List[str] = []
    for s in manual.sections:
        toc_items.append(f'    <li><a href="#{s.anchor}">{html.escape(s.heading)}</a>')
        if s.subsections:
            toc_items.append("      <ol>")
            for sub in s.subsections:
                toc_items.append(
                    f'        <li><a href="#{sub.anchor}">'
                    f"{html.escape(sub.heading)}</a></li>"
                )
            toc_items.append("      </ol>")
        toc_items.append("    </li>")

    sections_html: List[str] = []
    for section in manual.sections:
        parts = [f'  <section id="{section.anchor}">']
        parts.append(f"    <h2>{html.escape(section.heading)}</h2>")
        intro = render_body(section.intro_lines)
        if intro:
            parts.append(intro)
        for sub in section.subsections:
            parts.append(f'    <h3 id="{sub.anchor}">{html.escape(sub.heading)}</h3>')
            parts.append(render_body(sub.lines))
        parts.append("  </section>")
        sections_html.append("\n".join(parts))

    preamble = " ".join(s.strip() for s in manual.preamble_lines if s.strip())
    footer_text = " — ".join(manual.title_lines)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(doc_title)} — {html.escape(subtitle)}</title>
  <style>{CSS}</style>
</head>
<body>

<nav>
  <div class="nav-title">{html.escape(doc_title)}</div>
  <div class="nav-subtitle">{html.escape(subtitle)}</div>
  <h2>Contents</h2>
  <ol>
{chr(10).join(toc_items)}
  </ol>
</nav>

<header>
  <h1>{html.escape(doc_title)}</h1>
  <p>{html.escape(subtitle)}</p>
</header>

<div class="layout">
<main>

  <div class="preamble">{html.escape(preamble)}</div>

{chr(10).join(sections_html)}

</main>
</div>

<footer>{html.escape(footer_text)}</footer>

<script>{TOC_JS}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# JSON corpus
# ---------------------------------------------------------------------------


def build_corpus(manual: Manual) -> List[dict]:
    """Each section object carries a structured `id` (the subsection number)
    so downstream consumers (retrieval metadata, source checks) never have
    to parse headings."""
    entries: List[dict] = []
    if manual.preamble_text:
        entries.append(
            {
                "url": f"{HTML_URL}#top",
                "title": " — ".join(manual.title_lines),
                "sections": [
                    {"id": "0", "title": "Introduction", "text": manual.preamble_text}
                ],
            }
        )
    for section in manual.sections:
        sections: List[dict] = []
        intro = "\n".join(section.intro_lines).strip("\n")
        if intro:
            sections.append(
                {"id": str(section.number), "title": "Overview", "text": intro}
            )
        for sub in section.subsections:
            sections.append({"id": sub.number, "title": sub.title, "text": sub.text})
        entries.append(
            {
                "url": f"{HTML_URL}#{section.anchor}",
                "title": section.heading,
                "sections": sections,
            }
        )
    return entries


def main() -> None:
    text = TXT_PATH.read_text(encoding="utf-8")
    manual = parse_manual(text)

    n_subsections = sum(len(s.subsections) for s in manual.sections)
    print(f"Parsed {len(manual.sections)} sections, {n_subsections} subsections from {TXT_PATH.name}")

    HTML_PATH.write_text(render_html(manual), encoding="utf-8")
    print(f"Wrote {HTML_PATH}")

    corpus = build_corpus(manual)
    JSON_PATH.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_chunks = sum(len(e["sections"]) for e in corpus)
    print(f"Wrote {JSON_PATH} ({len(corpus)} entries, {n_chunks} chunk sections)")


if __name__ == "__main__":
    main()
