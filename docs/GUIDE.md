# RAG Guide

Architecture proof checklist for the Cognitive RAG Agent.

## Corpus

- **50 items** in `sandbox/research_papers/` with manifest at `corpus/MANIFEST.json`
- **6 dev notes** in `sandbox/rag_corpus/` (optional quick index)
- Download papers: `uv run python scripts/download_research_papers.py --count 50`
- Index into FAISS: `uv run python scripts/index_research_corpus.py` or Web UI **Index research corpus**

## Architecture gate

```bash
uv run python tests/test_architecture.py
```

Perception must stay **tool-blind** (zero MCP tool names in its SYSTEM prompt). Tool selection lives in Decision + MCP docstrings.

## Design deferrals (intentional)

This repo implements **dense RAG end-to-end** and defers production hardening to future releases:

1. **No hybrid retrieval** — dense FAISS only (no BM25 + RRF fusion yet)
2. **Heuristic chunking** — sliding window, not semantic boundaries
3. **FAISS reload on every read** — cross-process consistency over in-process cache
4. **Pinned embed model** — do not change `GEMINI_EMBED_MODEL` mid-project without re-indexing

Details and forward pointers: [`docs/DEFERRALS.md`](DEFERRALS.md) · API: `GET /api/deferrals`

## Eval queries

| Set | Count | Runner |
|-----|-------|--------|
| Base A–H | 8 scenarios (10 log files incl. C/F two-run) | `runs/eval_suite.py` or `runs/capture_queries.py` |
| Custom R1–R5 | 5 RAG queries (3 semantic recall) | `runs/eval_suite.py` |

**Custom queries**

| ID | Type | Question (abbrev.) |
|----|------|-------------------|
| R1 | semantic | SkillOpt external skill memory via scored rollouts |
| R2 | semantic | AutoResearchClaw — failed runs as feedback, not terminal errors |
| R3 | keyword | Sleep-like consolidation into fast weights (2605.26099) |
| R4 | cross-doc | Retrieval + generation for grounded QA |
| R5 | semantic | Info-Synth information-theoretic preference query synthesis |

Web UI: **RAG Queries** sidebar tab (loaded from `GET /api/queries/custom`). Each card shows kind badge and “requires index” note.

Full suite (architecture gate + A–H + R1–R5 indexed/no-corpus):

```bash
uv run python scripts/clean.py
uv run python runs/eval_suite.py
uv run python scripts/extract_traces.py
```

Logs: `logs/eval/` · Summary: `docs/SUBMISSION_TRACES.md`

## Submission artifacts

- GitHub repo with README
- Trace logs for A–H and R1–R5 (indexed vs no-corpus for custom queries)
- Web UI screenshots (`Images/`) for queries A–D
- Passing architecture gate
