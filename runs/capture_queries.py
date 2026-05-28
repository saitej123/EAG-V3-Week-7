#!/usr/bin/env python3
"""Run eight worked eval queries (A–H) and capture logs."""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def clean_workspace(*, keep_logs: bool = False) -> None:
    """Remove cache/temp dirs for a fresh run."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.clean import clean_workspace as _clean

    _clean(keep_logs=keep_logs)


def clean_state() -> None:
    state_dir = ROOT / "state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    print("Cleaned state/ directory.")


async def run_query(agent, query: str, log_filename: str) -> float:
    from loguru import logger

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / log_filename

    print(f"Running query: {query!r} -> {log_path.name}")
    sink_id = logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        level="DEBUG",
        encoding="utf-8",
        mode="w",
    )
    started = time.perf_counter()
    try:
        await agent.run(query)
    finally:
        logger.remove(sink_id)

    elapsed = time.perf_counter() - started
    print(f"Completed {log_filename} in {elapsed:.1f}s")
    return elapsed


async def main() -> None:
    clean_workspace(keep_logs=False)

    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    from cognitive_rag.agent import CognitiveAgent

    agent = CognitiveAgent()
    timings: list[tuple[str, float]] = []

    try:
        # --- A–D: web research and durable memory -----------------------------
        clean_state()
        timings.append(("query_a", await run_query(
            agent,
            "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.",
            "query_a.log",
        )))

        clean_state()
        timings.append(("query_b", await run_query(
            agent,
            "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.",
            "query_b.log",
        )))

        clean_state()
        timings.append(("query_c_run1", await run_query(
            agent,
            "My mom's birthday is 15 May 2026. Remember that and create reminders for two weeks before and on the day.",
            "query_c_run1.log",
        )))

        timings.append(("query_c_run2", await run_query(
            agent,
            "When is mom's birthday?",
            "query_c_run2.log",
        )))

        clean_state()
        timings.append(("query_d", await run_query(
            agent,
            "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.",
            "query_d.log",
        )))

        # --- E–H RAG queries ---------------------------------------------------
        clean_state()
        timings.append(("query_e", await run_query(
            agent,
            "Index the file papers/attention.md and tell me what the three key contributions of the Transformer architecture are according to this paper.",
            "query_e.log",
        )))

        clean_state()
        timings.append(("query_f_run1", await run_query(
            agent,
            "Index every .md file under papers/. Confirm how many chunks were indexed in total.",
            "query_f_run1.log",
        )))

        timings.append(("query_f_run2", await run_query(
            agent,
            "Across the papers I have indexed, what do they say about chain-of-thought reasoning?",
            "query_f_run2.log",
        )))

        timings.append(("query_g", await run_query(
            agent,
            "Across these papers, how do they handle the credit assignment problem?",
            "query_g.log",
        )))

        timings.append(("query_h", await run_query(
            agent,
            "Compare how the ReAct paper and the Chain-of-Thought paper differ in their treatment of intermediate reasoning.",
            "query_h.log",
        )))
    finally:
        await agent.action.aclose()

    total = sum(t for _, t in timings)
    print("\n--- Timing summary ---")
    for name, secs in timings:
        print(f"  {name}: {secs:.1f}s")
    print(f"  TOTAL: {total:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
