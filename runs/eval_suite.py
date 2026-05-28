#!/usr/bin/env python3
"""Eval suite: architecture gate, queries A–H, custom RAG queries, no-corpus baseline.

Logs land in logs/eval/ for README submission traces.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "eval"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Iteration ceilings (README budgets; hard cap AGENT_ITERATION_CEILING=4)
ITERATION_CEILINGS: dict[str, int] = {
    "base_a": 4,
    "base_b": 4,
    "base_c_run1": 4,
    "base_c_run2": 3,
    "base_d": 4,
    "base_e": 4,
    "base_f_run1": 4,
    "base_f_run2": 4,
    "base_g": 4,
    "base_h": 4,
}


def _iteration_count(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    import re

    hits = re.findall(r"─── iter (\d+)", log_path.read_text(encoding="utf-8", errors="replace"))
    return int(hits[-1]) if hits else None


# --- Eight base queries (verbatim from corpus/BASE_QUERIES.json) ----------------

def _base_queries() -> list[tuple[str, str, bool]]:
    from cognitive_rag.catalog import load_base_queries

    return load_base_queries()


BASE_QUERIES: list[tuple[str, str, bool]] = _base_queries()

# --- Five custom RAG queries (50-paper research corpus) -----------------------

def _custom_queries() -> list[tuple[str, str, str, list[str]]]:
    from cognitive_rag.catalog import load_custom_queries

    return load_custom_queries()


CUSTOM_QUERIES: list[tuple[str, str, str, list[str]]] = _custom_queries()


def clean_state() -> None:
    state = ROOT / "state"
    if state.exists():
        shutil.rmtree(state)
    from cognitive_rag.artifact_store import ensure_state_dirs

    ensure_state_dirs()


def clean_workspace(*, keep_logs: bool = True) -> None:
    for path in (ROOT / "state", ROOT / "sandbox_home", ROOT / ".crawl4ai", ROOT / "usage.json"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)
    if not keep_logs and LOGS.exists():
        shutil.rmtree(LOGS, ignore_errors=True)


async def run_query(
    agent, query: str, log_name: str, *, max_iterations: int | None = None
) -> float:
    from loguru import logger

    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / log_name
    print(f"  -> {log_name}")
    sink = logger.add(str(log_path), format="{message}", level="INFO", encoding="utf-8", mode="w")
    t0 = time.perf_counter()
    try:
        await agent.run(query, max_iterations=max_iterations)
    finally:
        logger.remove(sink)
    return time.perf_counter() - t0


def score_log(log_path: Path, must_contain: list[str]) -> tuple[bool, str]:
    if not log_path.is_file():
        return False, "log missing"
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    if ">>> FINAL ANSWER <<<" not in text and "final answer" not in text:
        return False, "no final answer marker"
    hits = [k for k in must_contain if k.lower() in text]
    if len(hits) < max(1, len(must_contain) // 2):
        return False, f"expected terms missing (got {hits})"
    return True, f"ok (matched {hits})"


async def main() -> None:
    clean_workspace(keep_logs=False)
    LOGS.mkdir(parents=True, exist_ok=True)

    # Architecture gate
    rc = subprocess.run([sys.executable, str(ROOT / "tests" / "test_architecture.py")], cwd=ROOT)
    if rc.returncode != 0:
        sys.exit(rc.returncode)

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from cognitive_rag.agent import CognitiveAgent

    agent = CognitiveAgent()
    summary: list[str] = []

    try:
        # Base A–H
        print("\n=== Base queries A–H ===")
        for key, query, clean in BASE_QUERIES:
            if clean:
                clean_state()
            elapsed = await run_query(agent, query, f"base_{key}.log")
            log_path = LOGS / f"base_{key}.log"
            iters = _iteration_count(log_path)
            ceiling = ITERATION_CEILINGS.get(f"base_{key}")
            iter_note = f", iters={iters}" if iters else ""
            if ceiling and iters and iters > ceiling:
                summary.append(f"base_{key}: {elapsed:.1f}s{iter_note} — FAIL exceeds ceiling {ceiling}")
            else:
                summary.append(f"base_{key}: {elapsed:.1f}s{iter_note}")

        # Custom RAG — with index
        print("\n=== Custom RAG (indexed corpus) ===")
        clean_state()
        from cognitive_rag.catalog import corpus_index_path
        from cognitive_rag.indexing import index_document_path

        corpus_path = corpus_index_path()
        corpus_dir = ROOT / "sandbox" / corpus_path
        md_files = sorted(corpus_dir.glob("*.md"))
        per_file: list[dict] = []
        total_chunks = 0
        for f in md_files:
            rel = f"{corpus_path}/{f.name}"
            result = index_document_path(rel, use_vlm=False)
            n = int(result.get("chunks_indexed", 0))
            total_chunks += n
            per_file.append({"path": rel, "chunks_indexed": n})
        idx = {
            "directory": corpus_path,
            "files_indexed": len(per_file),
            "chunks_indexed": total_chunks,
            "files": per_file,
        }
        print(f"  Indexed {corpus_path}: {idx['chunks_indexed']} chunks from {idx['files_indexed']} sidecars")
        (LOGS / "corpus_index_result.json").write_text(
            __import__("json").dumps(idx, indent=2), encoding="utf-8"
        )

        for key, kind, query, terms in CUSTOM_QUERIES:
            elapsed = await run_query(
                agent, query, f"custom_{key}_indexed.log", max_iterations=3
            )
            ok, msg = score_log(LOGS / f"custom_{key}_indexed.log", terms)
            summary.append(f"custom_{key}_indexed ({kind}): {elapsed:.1f}s — {msg}")

        # Custom RAG — no corpus baseline
        print("\n=== Custom RAG (no corpus baseline) ===")
        clean_state()
        for key, kind, query, terms in CUSTOM_QUERIES:
            elapsed = await run_query(
                agent, query, f"custom_{key}_no_corpus.log", max_iterations=3
            )
            ok_idx, _ = score_log(LOGS / f"custom_{key}_indexed.log", terms)
            ok_no, msg_no = score_log(LOGS / f"custom_{key}_no_corpus.log", terms)
            degraded = ok_idx and not ok_no
            summary.append(
                f"custom_{key}_no_corpus ({kind}): {elapsed:.1f}s — "
                f"{'PASS retrieval required' if degraded else 'CHECK manually: ' + msg_no}"
            )
    finally:
        await agent.action.aclose()

    print("\n=== Summary ===")
    for line in summary:
        print(line)
    print(f"\nLogs written to {LOGS}/")
    print("Paste excerpts into README or attach logs/eval/ for submission.")

    rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "extract_traces.py")], cwd=ROOT)
    if rc.returncode != 0:
        print("Warning: trace summary generation incomplete (some logs missing).")


if __name__ == "__main__":
    asyncio.run(main())
