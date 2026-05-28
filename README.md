<p align="center">
  <img src="Images/app-icon.svg" alt="Cognitive RAG Agent" width="96"/>
</p>

<h1 align="center">Cognitive RAG Agent</h1>

<p align="center">
  <strong>Memory → Perception → Decision → Action</strong><br/>
  Multi-source web synthesis · FAISS document RAG · MCP tool loop · FastAPI Web UI
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#demo">Demo</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#test-case-screenshots-web-ui">Test cases</a> ·
  <a href="#rag-query-screenshots-web-ui">RAG queries</a> ·
  <a href="#architecture-four-layers--loop">Architecture</a> ·
  <a href="#tests--evaluation">Tests</a>
</p>

A general-purpose cognitive agent with **durable vector memory (FAISS)**, **document RAG**, and **Pydantic v2** boundaries across a four-role MCP loop.

## Project layout

```
├── cognitive_rag/           # Agent core (loop, memory, MCP, indexing)
│   ├── agent.py             # Cognitive loop orchestration
│   ├── catalog.py           # Document manager API + BASE/QUERY spec loaders
│   ├── documents.py         # PDF normalize + VLM page extraction (batched, retry-safe)
│   ├── indexing.py          # Chunking, sidecar fast-path, index_document_path
│   ├── indexing_async.py    # Non-blocking index jobs for FastAPI + SSE progress
│   ├── llm_retry.py         # Retry/backoff for VLM, inference, embeddings
│   ├── memory.py, perception.py, decision.py, action.py, …
│   └── mcp_server.py        # MCP stdio server
├── app.py                   # FastAPI Web UI + document APIs + SSE logs
├── templates/index.html     # Chat UI + document manager
├── sandbox/
│   ├── research_papers/     # 50 arXiv PDFs + markdown sidecars
│   ├── papers/              # Sample `.md` paper summaries (+ optional PDFs)
│   ├── rag_corpus/          # 6 dev notes (incl. design deferrals)
│   └── uploads/             # User uploads (via document manager)
├── corpus/
│   ├── MANIFEST.json        # 50-paper research corpus manifest
│   ├── BASE_QUERIES.json    # Built-in test scenarios A–H
│   └── QUERY_SPEC.json      # Five custom RAG query specs
├── docs/DEFERRALS.md        # Intentional design simplifications
├── runs/eval_suite.py       # Test cases A–H + custom RAG queries
├── scripts/                 # corpus build, paper download, index helpers
├── Images/                  # App icon + Web UI screenshots (test cases, RAG queries)
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Quick start

**Requires:** `uv`, Python 3.12+, and **`GEMINI_API_KEY`** in `.env` (see [`.env.example`](.env.example)).

```bash
uv sync
cp .env.example .env
./scripts/serve.sh
```

Open **http://127.0.0.1:8080/** — chat, live console, document manager, **Test Cases** (A–H), and **RAG Queries**.

**First-time RAG setup:** load papers with `uv run python scripts/download_research_papers.py --from-disk`, then **Documents → Index all (50 papers)** (or `uv run python scripts/index_research_corpus.py`).

## Demo

[![Cognitive RAG Agent walkthrough](https://img.youtube.com/vi/rGVKVTlnZHs/0.jpg)](https://youtu.be/rGVKVTlnZHs)

**[Watch on YouTube →](https://youtu.be/rGVKVTlnZHs)**

## Overview

A **document RAG desktop app** with FastAPI Web UI, document manager, and durable **FAISS** vector memory. The agent combines **multi-source web research**, **local document indexing** (PDF + markdown sidecars), and a **four-role cognitive loop** (Memory → Perception → Decision → Action) over MCP stdio tools.

| Capability | Details |
|------------|---------|
| **Document RAG** | Index `sandbox/research_papers/` (50 arXiv PDFs + sidecars), papers in `sandbox/papers/`, uploads, and notes in `rag_corpus/` |
| **Web synthesis** | Shared fallback chain — Tavily → crawl4ai → Gemini live search → DuckDuckGo |
| **Durable memory** | FAISS + embeddings; facts and indexed chunks persist under `state/` |
| **Structured loop** | Pydantic v2 boundaries; smart iteration budget (default **3**, ceiling **4**); SSE live console |
| **Document manager** | Upload, per-doc index/reindex/delete, bulk index research corpus |
| **Resilient indexing** | VLM pages in batches of 10 with retry/backoff; markdown sidecar fast-path for PDFs |

### Indexing & retrieval

| Setting | Default |
|---------|---------|
| **Embedding model** | [`gemini-embedding-2`](https://ai.google.dev/gemini-api/docs/embeddings) (`GEMINI_EMBED_MODEL`; optional `GEMINI_EMBED_OUTPUT_DIM`: 768 / 1536 / 3072) |
| **Chunk size** | **400** characters, **80** overlap (sliding window) |
| **VLM long pages** | Sub-chunk at **1200** characters (`VLM_PAGE_CHUNK_SIZE`) |
| **Vector store** | FAISS + `state/memory.json` |

**How indexing works**

1. **Pick a file** under `sandbox/` (upload, research paper, or sidecar).
2. **Extract text** — `.md` / `.txt` read as UTF-8; PDFs with a matching `.md` sidecar use the sidecar (no VLM); other PDFs/images go through VLM page extraction (10 pages per batch).
3. **Chunk** — sliding window (400 / 80); each chunk becomes a memory fact with path + chunk metadata.
4. **Embed** — `gemini-embedding-2` vectors written to FAISS (`state/index.faiss`).
5. **Retrieve** — agent calls `search_knowledge`; `memory.read()` does dense search first, keyword fallback if vectors are unavailable.

Re-index after changing the embed model (`scripts/clean.py`, then bulk-index again).

### Research corpus ([`corpus/MANIFEST.json`](corpus/MANIFEST.json))

| Field | Value |
|-------|--------|
| **Name** | Research Paper RAG Corpus |
| **Items** | **50** arXiv papers (cs.LG / cs.CL / cs.AI) — PDF + `.md` sidecar each |
| **Location** | `sandbox/research_papers/` |
| **Index** | `index_directory('research_papers')` (UI: **Index all (50 papers)**) |
| **Retrieve** | `search_knowledge` |

Download if needed: `uv run python scripts/download_research_papers.py --from-disk` · index: `scripts/index_research_corpus.py`

**Tech:** Python 3.12 · FastAPI · FAISS · Gemini (LLM + embeddings) · MCP stdio · Tavily / crawl4ai / DuckDuckGo fallbacks

Screenshots: [Test cases](#test-case-screenshots-web-ui) · [RAG queries](#rag-query-screenshots-web-ui) · Module map: [Architecture](#architecture-four-layers--loop)

## Test case screenshots (Web UI)

Built-in scenarios **A–H** from the **Test Cases** sidebar ([`corpus/BASE_QUERIES.json`](corpus/BASE_QUERIES.json)).

### Query A — Shannon Wikipedia

> Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/t1_a.png" alt="Query A — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t1_b.png" alt="Query A — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### Query B — Tokyo activities + weather

> Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent trace (in chat)</strong></sub><br/><br/>
      <img src="Images/t2_a.png" alt="Query B — agent working" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t2_b.png" alt="Query B — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### Query D — Asyncio multi-source synthesis

> Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/t3_a.png" alt="Query D — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t3_b.png" alt="Query D — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### Query C — Mom's birthday (two runs)

**Run 1 — save facts**

> My mom's birthday is 15 May 2026. Remember that and create reminders for two weeks before and on the day.

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/t4_a_a.png" alt="Query C Run 1 — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t4_a_b.png" alt="Query C Run 1 — live console" width="98%"/>
    </td>
  </tr>
</table>

**Run 2 — recall** (same `state/`, do not clear between runs)

> When is mom's birthday?

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/t4_b_a.png" alt="Query C Run 2 — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t4_b_b.png" alt="Query C Run 2 — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### Query E — Index `papers/attention.md`

> Index the file papers/attention.md and tell me what the three key contributions of the Transformer architecture are according to this paper.

<table>
  <tr>
    <td align="center" width="33%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/t5_a.png" alt="Query E — agent result" width="98%"/>
    </td>
    <td align="center" width="33%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t5_b.png" alt="Query E — live console" width="98%"/>
    </td>
    <td align="center" width="33%" valign="top">
      <sub><strong>Document manager</strong></sub><br/><br/>
      <img src="Images/t5_c.png" alt="Query E — indexed in document manager" width="98%"/>
    </td>
  </tr>
</table>

---

### Query F — Bulk index `papers/` (two runs)

**Run 1 — index directory**

> Index every .md file under papers/. Confirm how many chunks were indexed in total.

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent (working)</strong></sub><br/><br/>
      <img src="Images/t6_a1.png" alt="Query F Run 1 — working" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/t6_a2.png" alt="Query F Run 1 — result" width="98%"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t6_a3.png" alt="Query F Run 1 — live console" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Document manager</strong></sub><br/><br/>
      <img src="Images/t6_a4.png" alt="Query F Run 1 — indexed papers" width="98%"/>
    </td>
  </tr>
</table>

**Run 2 — CoT recall** (reuse `state/`)

> Across the papers I have indexed, what do they say about chain-of-thought reasoning?

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/t6_b1.png" alt="Query F Run 2 — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t6_b2.png" alt="Query F Run 2 — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### Query G — Synonym recall (credit assignment)

> Across these papers, how do they handle the credit assignment problem?

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/t7_a.png" alt="Query G — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t7_b.png" alt="Query G — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### Query H — ReAct vs Chain-of-Thought

> Compare how the ReAct paper and the Chain-of-Thought paper differ in their treatment of intermediate reasoning.

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/t8_a.png" alt="Query H — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/t8_b.png" alt="Query H — live console" width="98%"/>
    </td>
  </tr>
</table>

## RAG query screenshots (Web UI)

Five custom corpus queries from the **RAG Queries** sidebar ([`corpus/QUERY_SPEC.json`](corpus/QUERY_SPEC.json)). Each row: verbatim query, then **agent result** (left) and **live console** (right).

### SkillOpt external skill memory

> In my indexed research library, which work treats an agent skill as trainable external memory updated through scored rollouts rather than weight fine-tuning?

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/Rag_images/r1.png" alt="SkillOpt — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/Rag_images/r1_a.png" alt="SkillOpt — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### AutoResearchClaw failure-as-signal

> Which indexed paper treats failed experiment executions as informative feedback rather than terminal errors in a multi-agent autonomous research workflow?

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/Rag_images/r2.png" alt="AutoResearchClaw — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/Rag_images/r2_a.png" alt="AutoResearchClaw — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### Sleep-like LM consolidation

> What do my indexed papers say about sleep-like consolidation that writes recent context into fast weights before clearing the KV cache?

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/Rag_images/r3.png" alt="Sleep-like consolidation — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/Rag_images/r3_a.png" alt="Sleep-like consolidation — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### Retrieval + generation synthesis

> Across my indexed arXiv library, how do recent works combine retrieval with generation for grounded question answering or detection?

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/Rag_images/r4.png" alt="Retrieval synthesis — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/Rag_images/r4_a.png" alt="Retrieval synthesis — live console" width="98%"/>
    </td>
  </tr>
</table>

---

### Info-Synth preference queries

> Which indexed study generates comparison questions using an information-theoretic selection criterion when obtaining trustworthy pairwise user labels is costly?

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <sub><strong>Agent result</strong></sub><br/><br/>
      <img src="Images/Rag_images/r5.png" alt="Info-Synth — agent result" width="98%"/>
    </td>
    <td align="center" width="50%" valign="top">
      <sub><strong>Live console</strong></sub><br/><br/>
      <img src="Images/Rag_images/r5_a.png" alt="Info-Synth — live console" width="98%"/>
    </td>
  </tr>
</table>

## Fresh start (clean runtime artifacts)

```bash
uv run python scripts/clean.py
```

Removes `state/`, `logs/`, caches, `usage.json`, and agent-generated sandbox files. **Keeps** `sandbox/research_papers/`, `sandbox/papers/`, `sandbox/rag_corpus/`, and `sandbox/uploads/`.

---

## Search & fetch providers (`search_providers.py`)

Single source of truth for external data. Used by **`cognitive_rag/mcp_server.py`**, **`cognitive_rag/action.py`** (direct fallback), and **`cognitive_rag/agent.py`** (emergency rescue).

| Tool | Provider order |
|------|----------------|
| **`web_search`** | **1. Tavily** → **2. crawl4ai** (DDG SERP crawl) → **3. Gemini live search** (Google grounding) → **4. DuckDuckGo** (library + httpx HTML scrape) |
| **`fetch_url` / `fetch_urls`** | **1. crawl4ai** (warm browser pool in MCP) → **2. httpx** plain fetch |

- Empty `web_search` args are auto-filled from the user query and active goal via **`enrich_tool_call()`**.
- Tavily/DDG usage is tracked in **`usage.json`** (monthly cap **950** calls on Tavily).
- MCP verbose traces (`[MCP] -->`, fetch previews) log at **DEBUG** (terminal only); iteration summary lines log at **INFO** (CLI + UI).

---

## On-disk state (`state/` — gitignored)

| Path | Role |
|------|------|
| **`state/memory.json`** | **`{"items": [ ... MemoryItem ... ]}`** (legacy **`{"facts":[]}`** is migrated on load) |
| **`state/commerce.db`** | Cached product rows (optional PDP catalog) |
| **`state/artifacts/`** | **`ArtifactStore`** blobs + metadata (and legacy **`art:*.txt`**) |

### Clean slate

```bash
uv run python scripts/clean.py
```

Or **Sidebar → Test Cases → Clear Durable State** (wipes `state/` only).

---

## MCP server

**`python -m cognitive_rag.mcp_server`** — stdio tools (**`web_search`**, **`fetch_url`**, **`fetch_urls`**, **`query_database`**, **`analyze_image_url`**, sandbox file tools, **`get_time`**, **`currency_convert`**, **`index_document`**, **`search_knowledge`**, …).

- **`web_search`**: delegates to **`search_providers.web_search_with_fallbacks()`** with Tavily/DDG usage tracking.
- **`fetch_url` / `fetch_urls`**: crawl4ai browser pool (parallel batch up to 3 URLs); httpx fallback on crawl failure.
- Requires matching **`uv`** deps (**`crawl4ai`**, **`tavily`**, **`duckduckgo-search`**, **`google-genai`**).

---

## Architecture (four layers + loop)

| Layer | Module | Responsibility |
|--------|--------|------------------|
| **Contracts** | **`cognitive_rag/schemas.py`** | `MemoryItem`, `Goal`, `Observation`, `ToolCall`, `DecisionOutput`, LLM payload models (`PerceptionLLMResponse`, `DecisionLLMFlat`, `MemoryClassifyLLM`, …), commerce DB rows |
| **Memory** | **`cognitive_rag/memory.py`** (`MemoryService` / `MemoryManager`) | **`read()`** — vector-first FAISS + keyword fallback; **`remember()`** / **`record_outcome()`** / **`add_fact()`**; MCP **`index_document`** / **`search_knowledge`**; **`query_products`** — SQLite catalog |
| **Artifacts** | **`cognitive_rag/artifact_store.py`** | **`ArtifactStore`** — content-addressable **`art:<sha256-prefix>`** (`.bin` + `.json`); reads legacy **`art:*.txt`**; large MCP results offload above **4 KiB** |
| **Perception** | **`cognitive_rag/perception.py`** | **`observe(query, hits, history, prior_goals, run_id) → Observation`** — ordered goals; model emits **`artifact_index`** only (no free-form `art:` handles) |
| **Decision** | **`cognitive_rag/decision.py`** | **`next_step(...) → DecisionOutput`** — **`answer`** *or* **`tool_call`** (wire format **`DecisionLLMFlat`** → mapped in code) |
| **Action** | **`cognitive_rag/action.py`** | **`execute(ToolCall, store, fallback_query=…) → (descriptor, artifact_id?)`** — MCP **stdio**; direct **`search_providers`** fallback when MCP fails; **`gemini_live_search`** for explicit INR shopping checks; blocks **`art:`** in tool arguments |
| **Search / fetch** | **`cognitive_rag/search_providers.py`** | **`web_search_with_fallbacks()`** (Tavily → crawl4ai → Gemini → DDG); **`httpx_plain_fetch`**; **`enrich_tool_call()`** auto-fills empty tool args from user query + goal |
| **Indexing** | **`cognitive_rag/indexing.py`** | Sidecar fast-path for PDFs, chunking, **`index_directory`** |
| **Async indexing** | **`cognitive_rag/indexing_async.py`** | **`asyncio.to_thread`** bulk jobs — SSE stays live during index |
| **LLM retry** | **`cognitive_rag/llm_retry.py`** | Retry/backoff for VLM, **`generate_content`**, **`generate_structured_with_retry`**, embeddings; truncated JSON salvage |
| **Documents** | **`cognitive_rag/documents.py`** | PDF → VLM (10 pages/batch, retry, placeholder on page failure) |
| **Catalog / UI API** | **`cognitive_rag/catalog.py`** | Document inventory, upload lifecycle, **`BASE_QUERIES.json`** + **`QUERY_SPEC.json`** loaders |
| **Config / timeouts** | **`cognitive_rag/llm_env.py`** | API keys, model list, **`shared_gemini_client()`**, **`gemini-embedding-2`** vectors (optional gateway), iteration budget (**default 3**, auto-extend up to **4**), **60s** LLM step timeout |
| **Orchestration** | **`cognitive_rag/agent.py`** | **`remember(user_query)`** once, then **`read → observe → next_step → execute → record_outcome`**; **`resolve_iteration_budget()`**; structured logs; early exit; emergency rescue; heuristic **`web_search`** after empty Decision or LLM timeout |

**Web UI:** **`app.py`** — FastAPI, **`GET /`**, **`POST /run-agent`**, **`GET /stream-logs`** (SSE; queue bound at lifespan startup). **Document manager** REST API under **`/api/documents/*`**. One agent run and one index job at a time → **429** if busy.

### Document manager (Web UI)

Open the **Documents** sidebar tab for a full-screen manager:

| Feature | Description |
|---------|-------------|
| **Stats** | Indexed docs, chunks, FAISS vectors, sandbox file counts |
| **Upload** | Drag-and-drop → `sandbox/uploads/` |
| **Table** | All sandbox + indexed docs with metadata, filter, search |
| **Actions** | Per-doc **Index**, **Reindex**, **Delete index** |
| **Bulk** | **Index all (50 papers)** — bulk-index `research_papers/` sidecars |

---

## Tests & evaluation

### Unit & architecture tests

```bash
uv sync --extra dev
uv run pytest tests/ -q
uv run python tests/test_architecture.py
```

Repo verification (query specs, manifest, trace layout):

```bash
uv run python scripts/check_submission.py
uv run pytest tests/test_submission_spec.py tests/test_architecture.py -q
```

### Batch runners

Custom RAG query specs live in [`corpus/QUERY_SPEC.json`](corpus/QUERY_SPEC.json).

```bash
uv run python scripts/clean.py              # fresh workspace (keeps corpus dirs)
uv run python runs/eval_suite.py            # test cases + custom RAG queries → logs/eval/
uv run python scripts/extract_traces.py     # → docs/SUBMISSION_TRACES.md
```

### Custom corpus queries

Verbatim queries and UI screenshots: [RAG query screenshots](#rag-query-screenshots-web-ui). Spec: [`corpus/QUERY_SPEC.json`](corpus/QUERY_SPEC.json).

**Eval contract:** index `research_papers/` sidecars first → each query **passes** with corpus; run again with empty `state/` → each query **fails or degrades**. Three queries require **semantic recall** (paraphrases without verbatim chunk overlap).

### Trace artifacts

| Artifact | Location |
|----------|----------|
| Eval logs | `logs/eval/base_*.log`, `custom_*_indexed.log`, `custom_*_no_corpus.log` |
| Trace summary | [`docs/SUBMISSION_TRACES.md`](docs/SUBMISSION_TRACES.md) |
| Corpus manifest | [`corpus/MANIFEST.json`](corpus/MANIFEST.json) |
| RAG guide | [`docs/GUIDE.md`](docs/GUIDE.md) |
| UI screenshots | [`Images/`](Images/) · [Test cases](#test-case-screenshots-web-ui) · [RAG queries](#rag-query-screenshots-web-ui) |
