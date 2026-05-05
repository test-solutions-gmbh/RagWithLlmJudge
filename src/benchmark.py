import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src._io import atomic_write_json


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_ROOT = REPO_ROOT / "data" / "benchmarks"


class BenchmarkError(Exception):
    """Raised for user-facing benchmark validation/IO problems."""


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _sanitise_for_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "model"


def _benchmark_dir(name: str) -> Path:
    return BENCHMARKS_ROOT / name


def _benchmark_file(name: str) -> Path:
    return _benchmark_dir(name) / "benchmark.json"


def _judge_runs_dir(name: str) -> Path:
    return _benchmark_dir(name) / "judge_runs"


def list_benchmarks() -> List[str]:
    if not BENCHMARKS_ROOT.exists():
        return []
    return sorted(
        p.name
        for p in BENCHMARKS_ROOT.iterdir()
        if p.is_dir() and (p / "benchmark.json").exists()
    )


def load_benchmark(name: str) -> Dict[str, Any]:
    path = _benchmark_file(name)
    if not path.exists():
        raise BenchmarkError(f"Benchmark '{name}' not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def promote_from_results(
    results_path: Path,
    name: str,
    *,
    force: bool = False,
) -> Path:
    if not results_path.exists():
        raise BenchmarkError(f"Results file not found: {results_path}")

    target = _benchmark_file(name)
    if target.exists() and not force:
        raise BenchmarkError(
            f"Benchmark '{name}' already exists at {target}. Pass force=True to overwrite."
        )

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    entries = results.get("entries", [])
    if not entries:
        raise BenchmarkError(f"Results file {results_path} contains no entries.")

    missing = [
        e.get("id", "<no-id>")
        for e in entries
        if e.get("human_eval", {}).get("acceptable_answer") is None
    ]
    if missing:
        joined = ", ".join(missing)
        raise BenchmarkError(
            "Cannot promote: the following entries have no human verdict yet: "
            f"{joined}"
        )

    frozen_entries = []
    for e in entries:
        frozen_entries.append(
            {
                "id": e["id"],
                "category": e.get("category"),
                "question": e["question"],
                "expected_answer": e["expected_answer"],
                "source_title": e.get("source_title"),
                "source_url": e.get("source_url"),
                "rag_response": e["rag_response"],
                "retrieved_contexts": e.get("retrieved_contexts", []),
                "human_eval": {
                    "acceptable_answer": bool(
                        e["human_eval"]["acceptable_answer"]
                    ),
                    "comment": e["human_eval"].get("comment"),
                },
            }
        )

    payload = {
        "benchmark_id": name,
        "source_run_id": results.get("run_id"),
        "rag_model": results.get("rag_model"),
        "created_at": _now_slug(),
        "entries": frozen_entries,
    }

    atomic_write_json(target, payload)
    return target


def save_judge_run(
    name: str,
    judge_model: str,
    verdicts: List[Dict[str, Any]],
) -> Path:
    if not _benchmark_file(name).exists():
        raise BenchmarkError(f"Benchmark '{name}' does not exist.")

    timestamp = _now_slug()
    filename = f"{_sanitise_for_filename(judge_model)}__{timestamp}.json"
    path = _judge_runs_dir(name) / filename
    payload = {
        "benchmark_id": name,
        "judge_model": judge_model,
        "created_at": timestamp,
        "verdicts": verdicts,
    }
    atomic_write_json(path, payload)
    return path


def load_judge_runs(name: str) -> List[Dict[str, Any]]:
    runs_dir = _judge_runs_dir(name)
    if not runs_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_filename"] = path.name
        out.append(data)
    return out


def compute_agreement(
    benchmark: Dict[str, Any],
    judge_run: Dict[str, Any],
) -> Dict[str, Any]:
    judge_by_id: Dict[str, Dict[str, Any]] = {
        v["id"]: v for v in judge_run.get("verdicts", [])
    }

    agree = disagree = pending = 0
    per_question: List[Dict[str, Any]] = []
    for entry in benchmark.get("entries", []):
        h: Optional[bool] = entry["human_eval"]["acceptable_answer"]
        verdict = judge_by_id.get(entry["id"])
        j: Optional[bool] = verdict["acceptable_answer"] if verdict else None
        reason = verdict.get("reason") if verdict else None

        if j is None or h is None:
            pending += 1
            agreement = "pending"
        elif j == h:
            agree += 1
            agreement = "agree"
        else:
            disagree += 1
            agreement = "disagree"

        per_question.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "human": h,
                "judge": j,
                "reason": reason,
                "agreement": agreement,
            }
        )

    total = len(benchmark.get("entries", []))
    decided = agree + disagree
    alignment_pct = (agree / decided * 100.0) if decided else 0.0

    return {
        "agree": agree,
        "disagree": disagree,
        "pending": pending,
        "total": total,
        "alignment_pct": alignment_pct,
        "per_question": per_question,
    }
