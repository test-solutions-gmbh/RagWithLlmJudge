import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src import benchmark as bench


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "data" / "results"
LOGO_PATH = REPO_ROOT / "assets" / "ts-logo-transparent.png"
FAVICON_PATH = REPO_ROOT / "assets" / "ts-lean-transparent.png"

CI_BLUE_1 = "#5383C6"
CI_GREEN_1 = "#C5E784"
CI_RED = "#D82953"
CI_GREY = "#AEA9A2"
CI_GREY_DARK = "#6B6560"
CI_BLACK = "#000000"

_CUSTOM_CSS = """
<style>
textarea { background-color: #E8E8E8 !important; }
</style>
"""


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-path",
        default=None,
        help="Path to the results.json this deployment is bound to.",
    )
    known, _unknown = parser.parse_known_args()
    return known


def _resolve_results_path() -> Path:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    args = _parse_cli()
    if args.results_path:
        return Path(args.results_path).resolve()
    env_path = os.getenv("RESULTS_PATH")
    if env_path:
        return Path(env_path).resolve()
    if not RESULTS_ROOT.exists():
        st.error(
            f"No results directory at `{RESULTS_ROOT.relative_to(REPO_ROOT)}`. "
            "Run `python -m src.cli generate-rag-responses` first, or pass "
            "`-- --results-path <file>` when launching streamlit."
        )
        st.stop()
    runs = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()])
    if not runs:
        st.error(f"No result runs in `{RESULTS_ROOT.relative_to(REPO_ROOT)}`.")
        st.stop()
    return runs[-1] / "results.json"


def _load_results(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _verdict_icon(value: Optional[bool]) -> str:
    if value is True:
        return "Acceptable"
    if value is False:
        return "Not acceptable"
    return "Pending"


def _verdict_button_css(verdict: Optional[bool]) -> str:
    if verdict is True:
        color, hover = CI_BLUE_1, "#4272B5"
    elif verdict is False:
        color, hover = CI_RED, "#C12248"
    else:
        return ""
    return (
        f'<style>.stButton > button[kind="primary"]'
        f"{{background-color:{color};border-color:{color};color:#fff}}"
        f'.stButton > button[kind="primary"]:hover'
        f"{{background-color:{hover};border-color:{hover};color:#fff}}"
        f"</style>"
    )


def _verdict_badge(value: Optional[bool]) -> str:
    if value is True:
        return (
            f'<span style="background:{CI_BLUE_1};color:#fff;'
            "padding:2px 10px;border-radius:10px;font-size:0.85rem;"
            'font-weight:600">Acceptable</span>'
        )
    if value is False:
        return (
            f'<span style="background:{CI_RED};color:#fff;'
            "padding:2px 10px;border-radius:10px;font-size:0.85rem;"
            'font-weight:600">Not acceptable</span>'
        )
    return (
        f'<span style="background:{CI_GREY_DARK};color:#fff;'
        "padding:2px 10px;border-radius:10px;font-size:0.85rem;"
        'font-weight:600">Pending</span>'
    )


_LATEX_TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
_LATEX_DELIM_RE = re.compile(r"\\[\(\)\[\]]")
_LATEX_COMMAND_REPLACEMENTS = [
    (re.compile(r"\\times\b"), "×"),
    (re.compile(r"\\cdot\b"), "·"),
    (re.compile(r"\\div\b"), "÷"),
    (re.compile(r"\\approx\b"), "≈"),
    (re.compile(r"\\neq\b"), "≠"),
    (re.compile(r"\\leq\b"), "≤"),
    (re.compile(r"\\geq\b"), "≥"),
    (re.compile(r"\\le\b"), "≤"),
    (re.compile(r"\\ge\b"), "≥"),
    (re.compile(r"\\pm\b"), "±"),
    (re.compile(r"\\to\b"), "→"),
    (re.compile(r"\\Rightarrow\b"), "⇒"),
    (re.compile(r"\\rightarrow\b"), "→"),
]


def _strip_latex(text: str) -> str:
    """Convert the bits of LaTeX that the RAG model occasionally emits
    (`\\(...\\)` math spans, `\\text{...}`, `\\times`, `\\approx`, ...) into
    plain text with Unicode operators. Streamlit's markdown renderer only
    understands `$...$` math, so without this the formulas would render as
    literal backslash gibberish."""
    text = _LATEX_DELIM_RE.sub("", text)
    text = _LATEX_TEXT_RE.sub(r"\1", text)
    for pattern, replacement in _LATEX_COMMAND_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def _md_safe(text: Optional[str]) -> str:
    """Escape characters that Streamlit's markdown renderer would
    otherwise interpret. Notably `$` triggers inline LaTeX, which
    silently swallows whitespace and italicizes everything in between
    (e.g. answers that mention `$200/month` and `$3,000`).

    We also strip the LaTeX-style math the RAG model occasionally emits
    before escaping, so `\\(5 \\text{years} \\times 2\\)` renders as
    `5 years × 2` instead of a wall of literal backslashes."""
    if text is None:
        return ""
    text = _strip_latex(text)
    return text.replace("\\", "\\\\").replace("$", "\\$")


def _render_sidebar(payload: dict, results_path: Path) -> str:
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)

        st.divider()
        st.markdown("### Deployment")

        def _info_row(label: str, value: str) -> None:
            st.markdown(
                f"{label}: "
                f'<span style="color:{CI_BLACK};font-weight:400">{value}</span>',
                unsafe_allow_html=True,
            )

        _info_row("run_id", payload["run_id"])
        _info_row("rag_model", payload["rag_model"])
        _info_row("judge_model", payload.get("judge_model") or "not yet run")

        st.divider()
        st.markdown("### Progress")
        total_questions = len(payload["entries"])
        human_evaluated = sum(
            1
            for e in payload["entries"]
            if e["human_eval"]["acceptable_answer"] is not None
        )
        progress_ratio = (human_evaluated / total_questions) if total_questions else 0.0
        st.progress(
            progress_ratio,
            text=f"{human_evaluated}/{total_questions} questions evaluated",
        )

        st.divider()
        view = st.radio("View", ["Questions", "Summary", "Benchmark"], index=0)
        if st.button("Reload from disk"):
            st.rerun()
    return view


def _render_question(payload: dict, results_path: Path) -> None:
    entries = payload["entries"]
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = max(0, min(st.session_state.idx, len(entries) - 1))
    idx = st.session_state.idx
    entry = entries[idx]

    col_prev, col_title, col_next = st.columns([1, 6, 1])
    with col_prev:
        if st.button("◀ Prev", disabled=idx == 0, use_container_width=True):
            st.session_state.idx -= 1
            st.rerun()
    with col_title:
        st.markdown(
            f"### Question {idx + 1} of {len(entries)} · "
            f"`{entry.get('category') or 'uncategorized'}`"
        )
    with col_next:
        if st.button(
            "Next ▶",
            disabled=idx == len(entries) - 1,
            use_container_width=True,
        ):
            st.session_state.idx += 1
            st.rerun()

    st.markdown(f"**{_md_safe(entry['question'])}**")
    st.caption(f"Source: [{entry['source_title']}]({entry['source_url']})")

    col_gt, col_rag = st.columns(2)
    with col_gt:
        st.markdown("#### Expected answer")
        st.success(_md_safe(entry["expected_answer"]))
    with col_rag:
        st.markdown("#### RAG response")
        st.info(_md_safe(entry["rag_response"]))

    with st.expander("Retrieved contexts"):
        for i, ctx in enumerate(entry.get("retrieved_contexts", [])):
            st.markdown(f"**Context {i + 1}**")
            st.text(ctx)

    st.divider()
    st.markdown("### Human verdict")

    human_val = entry["human_eval"]["acceptable_answer"]
    st.markdown(_verdict_button_css(human_val), unsafe_allow_html=True)
    col_ok, col_bad, col_clear = st.columns(3)
    with col_ok:
        if st.button(
            "Acceptable",
            key=f"ok_{idx}",
            type="primary" if human_val is True else "secondary",
            use_container_width=True,
        ):
            entry["human_eval"]["acceptable_answer"] = True
            _save_results(results_path, payload)
            st.rerun()
    with col_bad:
        if st.button(
            "Not acceptable",
            key=f"bad_{idx}",
            type="primary" if human_val is False else "secondary",
            use_container_width=True,
        ):
            entry["human_eval"]["acceptable_answer"] = False
            _save_results(results_path, payload)
            st.rerun()
    with col_clear:
        if st.button("Clear", key=f"clr_{idx}", use_container_width=True):
            entry["human_eval"]["acceptable_answer"] = None
            entry["human_eval"]["comment"] = None
            _save_results(results_path, payload)
            st.rerun()

    comment = st.text_area(
        "Comment (optional)",
        value=entry["human_eval"].get("comment") or "",
        key=f"cmt_{idx}",
        height=80,
    )
    if st.button("Save comment", key=f"save_cmt_{idx}"):
        entry["human_eval"]["comment"] = comment.strip() or None
        _save_results(results_path, payload)
        st.toast("Comment saved")

    st.divider()
    st.markdown("### LLM judge verdict")
    if "revealed_judge" not in st.session_state:
        st.session_state.revealed_judge = {}
    revealed_for_question = st.session_state.revealed_judge.get(entry["id"], False)

    if not revealed_for_question:
        if st.button("Reveal LLM judge verdict", key=f"reveal_judge_{idx}"):
            st.session_state.revealed_judge[entry["id"]] = True
            st.rerun()
        st.info("LLM judge verdict is hidden until you reveal it.")
    else:
        judge = entry["llm_judge"]
        if judge["acceptable_answer"] is None:
            st.warning(
                "Judge has not run on this entry yet. Run "
                "`python -m src.cli evaluate-responses-with-llm-judge` in a terminal, then click "
                "**Reload from disk** in the sidebar."
            )
        else:
            st.markdown(
                _verdict_badge(judge["acceptable_answer"]), unsafe_allow_html=True
            )
            if judge.get("reason"):
                st.caption(_md_safe(judge["reason"]))

        if judge["acceptable_answer"] is not None and human_val is not None:
            if judge["acceptable_answer"] == human_val:
                st.success("Human and LLM judge agree")
            else:
                st.error("Human and LLM judge disagree")


def _render_summary(payload: dict) -> None:
    entries = payload["entries"]
    st.markdown("### Summary: human vs LLM judge")

    agree = disagree = pending = 0
    rows = []
    for i, e in enumerate(entries):
        h = e["human_eval"]["acceptable_answer"]
        j = e["llm_judge"]["acceptable_answer"]
        if h is None or j is None:
            pending += 1
            agreement = "—"
        elif h == j:
            agree += 1
            agreement = "agree"
        else:
            disagree += 1
            agreement = "disagree"
        rows.append(
            {
                "Q": f"Q{i + 1}",
                "Category": e.get("category") or "",
                "Human": _verdict_icon(h),
                "LLM judge": _verdict_icon(j),
                "Agreement": agreement,
            }
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Agree", agree)
    m2.metric("Disagree", disagree)
    m3.metric("Pending", pending)

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Per-question detail")
    for i, e in enumerate(entries):
        j = e["llm_judge"]
        h = e["human_eval"]
        with st.expander(f"Q{i + 1}: {e['question']}"):
            st.markdown(f"**Expected:** {_md_safe(e['expected_answer'])}")
            st.markdown(f"**RAG:** {_md_safe(e['rag_response'])}")
            st.markdown(
                f"**Human:** {_verdict_badge(h['acceptable_answer'])} "
                f"— {_md_safe(h.get('comment')) or '_no comment_'}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Judge:** {_verdict_badge(j['acceptable_answer'])} "
                f"— {_md_safe(j.get('reason')) or '_no reason_'}",
                unsafe_allow_html=True,
            )


def _render_benchmark() -> None:
    st.markdown("### Benchmark: judge alignment vs frozen human verdicts")
    st.caption(
        "Compare one or more judge runs against a benchmark's human verdicts. "
        "RAG responses and human verdicts are frozen; the judge model varies."
    )

    names = bench.list_benchmarks()
    if not names:
        try:
            rel = bench.BENCHMARKS_ROOT.relative_to(REPO_ROOT)
        except ValueError:
            rel = bench.BENCHMARKS_ROOT
        st.info(
            f"No benchmarks found under `{rel}`. Promote a results file with "
            "`python -m src.cli promote-benchmark --path <results.json> --name <slug>`."
        )
        return

    selected = st.selectbox("Benchmark", names, index=0)
    try:
        benchmark = bench.load_benchmark(selected)
    except bench.BenchmarkError as exc:
        st.error(str(exc))
        return

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Entries", len(benchmark.get("entries", [])))
    h2.metric("RAG model", benchmark.get("rag_model") or "—")
    h3.metric("Source run", benchmark.get("source_run_id") or "—")
    h4.metric("Created", benchmark.get("created_at") or "—")

    judge_runs = bench.load_judge_runs(selected)
    if not judge_runs:
        st.warning(
            "No judge runs yet for this benchmark. Run "
            f"`python -m src.cli judge-benchmark --name {selected}` "
            "(optionally with `--model <model>`)."
        )
        return

    labels = [f"{r['judge_model']} · {r['created_at']}" for r in judge_runs]
    label_to_run = dict(zip(labels, judge_runs))
    chosen_labels = st.multiselect("Judge runs to compare", labels, default=labels)
    chosen_runs = [label_to_run[label] for label in chosen_labels]
    if not chosen_runs:
        st.info("Select at least one judge run.")
        return

    summaries = [(run, bench.compute_agreement(benchmark, run)) for run in chosen_runs]

    rows = []
    for run, summary in summaries:
        rows.append(
            {
                "Judge model": run["judge_model"],
                "Created": run["created_at"],
                "Agree": summary["agree"],
                "Disagree": summary["disagree"],
                "Pending": summary["pending"],
                "Alignment %": f"{summary['alignment_pct']:.1f}",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Per-question detail")
    entries = benchmark.get("entries", [])
    for i, entry in enumerate(entries):
        h = entry["human_eval"]["acceptable_answer"]
        with st.expander(f"Q{i + 1}: {entry['question']}"):
            st.markdown(f"**Expected:** {_md_safe(entry['expected_answer'])}")
            st.markdown(f"**RAG:** {_md_safe(entry['rag_response'])}")
            st.markdown(
                f"**Human:** {_verdict_badge(h)} "
                f"— {_md_safe(entry['human_eval'].get('comment')) or '_no comment_'}",
                unsafe_allow_html=True,
            )
            for run, summary in summaries:
                per_q = next(
                    (q for q in summary["per_question"] if q["id"] == entry["id"]),
                    None,
                )
                if per_q is None:
                    continue
                tag = {
                    "agree": "agree",
                    "disagree": "disagree",
                    "pending": "pending",
                }[per_q["agreement"]]
                st.markdown(
                    f"**Judge ({run['judge_model']}):** "
                    f"{_verdict_badge(per_q['judge'])} {tag} "
                    f"— {_md_safe(per_q.get('reason')) or '_no reason_'}",
                    unsafe_allow_html=True,
                )


def main() -> None:
    st.set_page_config(
        page_title="LLM Judge Calibration",
        page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else None,
        layout="wide",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    results_path = _resolve_results_path()
    if not results_path.exists():
        st.error(f"Results file not found: `{results_path}`")
        st.stop()

    payload = _load_results(results_path)

    st.title("LLM Judge Calibration")

    view = _render_sidebar(payload, results_path)
    if view == "Questions":
        _render_question(payload, results_path)
    elif view == "Summary":
        _render_summary(payload)
    else:
        _render_benchmark()


if __name__ == "__main__":
    main()
else:
    # Streamlit imports the module rather than running it as __main__.
    main()
