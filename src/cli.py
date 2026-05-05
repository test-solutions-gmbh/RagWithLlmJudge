import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import APIConnectionError, AuthenticationError

from src import benchmark as bench
from src._io import atomic_write_json
from src.judge import build_judge_llm, judge_response
from src.rag import answer_question, build_rag_graph


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_ROOT = DATA_DIR / "results"
GROUND_TRUTHS_PATH = DATA_DIR / "ground_truths.json"

DEFAULT_RAG_MODEL = "openai/gpt-oss-120b:free"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b:free"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def _timestamp_slug() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return now


def _load_env() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path=dotenv_path, override=True)
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        os.environ[var] = "false"
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key or key.startswith("<") or key.endswith(">"):
        sys.exit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")


def _handbook_path() -> str:
    path = os.getenv("HANDBOOK_SOURCE", "basecamp_handbook.json")
    if not os.path.isabs(path):
        path = str(REPO_ROOT / path)
    return path


def _latest_results_file() -> Path:
    if not RESULTS_ROOT.exists():
        sys.exit(f"No results directory found at {RESULTS_ROOT}. Run generate-rag-responses first.")
    runs = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()])
    if not runs:
        sys.exit(f"No result runs inside {RESULTS_ROOT}. Run generate-rag-responses first.")
    latest = runs[-1] / "results.json"
    if not latest.exists():
        sys.exit(f"Expected results.json in {runs[-1]} but it is missing.")
    return latest


def cmd_generate_rag(_args) -> None:
    _load_env()
    handbook_path = _handbook_path()
    rag_model = os.getenv("RAG_MODEL") or DEFAULT_RAG_MODEL
    embedding_model = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()

    if not GROUND_TRUTHS_PATH.exists():
        sys.exit(f"Ground truths not found at {GROUND_TRUTHS_PATH}.")
    with open(GROUND_TRUTHS_PATH, "r", encoding="utf-8") as f:
        ground_truths = json.load(f)

    print(f"Building RAG graph from {handbook_path} with {rag_model}...")
    graph = build_rag_graph(
        handbook_path,
        llm_model=rag_model,
        embedding_model=embedding_model,
        openrouter_api_key=openrouter_key,
    )

    entries = []
    for gt in ground_truths:
        question = gt["question"]
        print(f"  [{gt['id']}] {question}")
        out = answer_question(graph, question)
        entries.append({
            "id": gt["id"],
            "category": gt.get("category"),
            "question": question,
            "expected_answer": gt["expected_answer"],
            "source_title": gt["source_title"],
            "source_url": gt["source_url"],
            "rag_response": out["response"],
            "retrieved_contexts": out["retrieved_contexts"],
            "llm_judge": {"acceptable_answer": None, "reason": None},
            "human_eval": {"acceptable_answer": None, "comment": None},
        })

    run_dir = RESULTS_ROOT / _timestamp_slug()
    results_path = run_dir / "results.json"
    payload = {
        "run_id": run_dir.name,
        "rag_model": rag_model,
        "judge_model": None,
        "entries": entries,
    }
    atomic_write_json(results_path, payload)
    print(f"\nWrote {len(entries)} entries to {results_path}")


def cmd_run_judge(args) -> None:
    _load_env()
    judge_model = os.getenv("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    results_path = Path(args.path) if args.path else _latest_results_file()
    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    print(f"Judging {len(payload['entries'])} entries in {results_path} with {judge_model}...")
    llm = build_judge_llm(judge_model, openrouter_api_key=openrouter_key)
    for entry in payload["entries"]:
        verdict = judge_response(
            question=entry["question"],
            expected_answer=entry["expected_answer"],
            response=entry["rag_response"],
            llm=llm,
        )
        entry["llm_judge"] = verdict
        flag = verdict["acceptable_answer"]
        tag = "PASS" if flag is True else ("FAIL" if flag is False else "ERR ")
        print(f"  [{entry['id']}] {tag} — {verdict['reason']}")

    payload["judge_model"] = judge_model
    atomic_write_json(results_path, payload)
    print(f"\nUpdated {results_path}")


def cmd_promote_benchmark(args) -> None:
    results_path = Path(args.path).resolve()
    try:
        target = bench.promote_from_results(
            results_path, args.name, force=args.force
        )
    except bench.BenchmarkError as exc:
        sys.exit(str(exc))
    print(f"Promoted {results_path} → {target}")


def cmd_judge_benchmark(args) -> None:
    _load_env()
    judge_model = args.model or os.getenv("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    try:
        benchmark = bench.load_benchmark(args.name)
    except bench.BenchmarkError as exc:
        sys.exit(str(exc))

    entries = benchmark["entries"]
    print(f"Judging {len(entries)} entries in benchmark '{args.name}' with {judge_model}...")
    llm = build_judge_llm(judge_model, openrouter_api_key=openrouter_key)

    verdicts = []
    for entry in entries:
        verdict = judge_response(
            question=entry["question"],
            expected_answer=entry["expected_answer"],
            response=entry["rag_response"],
            llm=llm,
        )
        verdicts.append(
            {
                "id": entry["id"],
                "acceptable_answer": verdict["acceptable_answer"],
                "reason": verdict["reason"],
            }
        )
        flag = verdict["acceptable_answer"]
        tag = "PASS" if flag is True else ("FAIL" if flag is False else "ERR ")
        print(f"  [{entry['id']}] {tag} — {verdict['reason']}")

    path = bench.save_judge_run(args.name, judge_model, verdicts)
    summary = bench.compute_agreement(
        benchmark, {"verdicts": verdicts}
    )
    print(f"\nWrote judge run to {path}")
    print(
        f"Alignment vs human benchmark: "
        f"{summary['agree']} agree / {summary['disagree']} disagree / "
        f"{summary['pending']} pending "
        f"({summary['alignment_pct']:.1f}% on decided entries)"
    )


def cmd_list_benchmarks(_args) -> None:
    names = bench.list_benchmarks()
    if not names:
        print(
            f"No benchmarks found under {bench.BENCHMARKS_ROOT}. "
            "Run promote-benchmark first."
        )
        return
    for name in names:
        try:
            data = bench.load_benchmark(name)
        except bench.BenchmarkError:
            continue
        runs = bench.load_judge_runs(name)
        rag_model = data.get("rag_model") or "unknown"
        print(
            f"- {name}  "
            f"(rag_model={rag_model}, "
            f"entries={len(data.get('entries', []))}, "
            f"judge_runs={len(runs)})"
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="llm-judge-demo",
        description="Run the RAG model and LLM judge over the curated ground-truth set.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-rag-responses", help="Run the RAG model over all ground-truth questions.")
    p_gen.set_defaults(func=cmd_generate_rag)

    p_judge = sub.add_parser("evaluate-responses-with-llm-judge", help="Run the LLM judge over an existing results file.")
    p_judge.add_argument(
        "--path",
        default=None,
        help="Path to results.json to judge. Defaults to the most recent run in data/results/.",
    )
    p_judge.set_defaults(func=cmd_run_judge)

    p_promote = sub.add_parser(
        "promote-benchmark",
        help="Freeze a fully human-evaluated results.json as a reusable benchmark.",
    )
    p_promote.add_argument(
        "--path",
        required=True,
        help="Path to the results.json to promote.",
    )
    p_promote.add_argument(
        "--name",
        required=True,
        help="Slug for the benchmark folder under data/benchmarks/.",
    )
    p_promote.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing benchmark with the same name.",
    )
    p_promote.set_defaults(func=cmd_promote_benchmark)

    p_judge_bench = sub.add_parser(
        "judge-benchmark",
        help="Run the LLM judge against a frozen benchmark and append a judge run.",
    )
    p_judge_bench.add_argument(
        "--name",
        required=True,
        help="Benchmark slug (folder name under data/benchmarks/).",
    )
    p_judge_bench.add_argument(
        "--model",
        default=None,
        help="Judge model to use. Defaults to $JUDGE_MODEL or openai/gpt-oss-120b:free.",
    )
    p_judge_bench.set_defaults(func=cmd_judge_benchmark)

    p_list = sub.add_parser(
        "list-benchmarks",
        help="List benchmarks under data/benchmarks/ and how many judge runs each has.",
    )
    p_list.set_defaults(func=cmd_list_benchmarks)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except AuthenticationError:
        sys.exit("OpenRouter rejected OPENROUTER_API_KEY (invalid or revoked). Check your .env file.")
    except APIConnectionError as exc:
        sys.exit(f"Could not reach the OpenRouter API: {exc}")


if __name__ == "__main__":
    main()
