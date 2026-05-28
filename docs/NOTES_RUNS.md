# Worked query notes

Eight queries exercise the RAG implementation. Run in order via:

```bash
uv run python runs/capture_queries.py
```

Or individually after cleaning `state/` where noted:

```bash
uv run python -m cognitive_rag.agent "Your query here"
```

## Web and memory queries (A–D)

| Query | Clean state first? | What it verifies |
|-------|-------------------|------------------|
| **A** Shannon Wikipedia | Yes | Artifact attach; vector uninvolved |
| **B** Tokyo + weather | Yes | Multi-goal; keyword memory path |
| **C run 1** Mom's birthday | Yes | `remember` + embeddings + reminders |
| **C run 2** When is birthday? | **No** (same `state/`) | FAISS cross-run recall |
| **D** Asyncio synthesis | Yes | Multi-source fetch + attach |

## Queries E–H (RAG)

| Query | Clean state first? | What it verifies |
|-------|-------------------|------------------|
| **E** Index attention.md | Yes | `index_document` + `search_knowledge` |
| **F run 1** Index all papers | Yes | Discovery + five `index_document` calls |
| **F run 2** CoT across corpus | **No** | Cold process reads persisted FAISS |
| **G** Credit assignment | **No** | Vector beats keyword (phrase absent from corpus) |
| **H** ReAct vs CoT | **No** | Cross-document synthesis |

## Expected disk state after F run 1

After **Query F run 1** completes, `state/` contains three files:

```text
state/
  memory.json        24 items: 15 chunk facts, 7 tool_outcomes, 2 classifier facts
  index.faiss        23 vectors at dimension 768
  index_ids.json     23 identifier strings, one per row of the FAISS index
```

| File | Role |
|------|------|
| `memory.json` | Source of truth — every `MemoryItem` row (with or without an embedding) |
| `index.faiss` | Dense vector index for semantic search |
| `index_ids.json` | Parallel id list: row *i* in FAISS maps to `index_ids[i]` in `memory.json` |

### Why 24 items but 23 vectors?

The discrepancy is intentional. The user query on F run 1 passes through `memory.remember`, which may classify part of the utterance as **`scratchpad`**. Scratchpad items are persisted to `memory.json` but **skip embedding** by design. The remaining 23 items all carry vectors and appear in the FAISS index.

Clearing agent state means deleting all three files (Web UI **Clear Durable State**, or `rm -rf state/`).

Chunk descriptors look like:

```text
[sandbox:papers/attention.md chunk 2/3]
```

**F run 2 → H** reuse this index without re-indexing.

## State on disk and what crosses the process boundary

The persistence story is what enables **Query F run 2**: a fresh agent process with no in-process memory still answers from the corpus indexed in run 1.

### Who writes what

| Writer | Typical writes |
|--------|----------------|
| **Agent process** | `memory.remember` (classifier facts, optional scratchpad), `memory.record_outcome` (tool_outcome rows) |
| **MCP subprocess** | `index_document` → `add_fact` (chunk facts), via Decision → Action |

Both writers append to the same `memory.json` and FAISS files synchronously before returning.

### The cross-process contract

Whatever lives on disk after a write is what the next read sees.

The agent's `memory.read()` **reloads `memory.json` and the FAISS index from disk on every call**. Caching the FAISS index in the agent process would hide MCP-subprocess writes from `index_document`. The implementation trades a few milliseconds of disk I/O per read for cross-process consistency without locks or shared memory.

A process that has never run before but starts with a populated `state/` directory works correctly: iteration one reads memory off disk. **Query F run 2** exercises this property.

## Honest design choices

These simplifications are deliberate; each has a forward pointer to a future upgrade. **Canonical reference:** [`docs/DEFERRALS.md`](DEFERRALS.md).

1. **Dense retrieval only** — no hybrid sparse partner (BM25 / learned-sparse + Reciprocal Rank Fusion). Production systems combine both; hybrid retrieval arrives later.

2. **Heuristic chunking** — `index_document` uses a fixed sliding window (default 400 words, 80 overlap) at arbitrary boundaries. Semantic chunking (LLM-aware sentence/paragraph breaks) is deferred to a future release.

3. **FAISS reload on every read** — acceptable at demo scale. At higher scale, memory-mapped indexes with mtime invalidation or inter-process locks become appropriate. This project does not model that scale.

4. **Fixed embedding model** — all vectors must live in the same semantic space. The gateway / `GEMINI_EMBED_MODEL` (default **`gemini-embedding-2`**) must stay consistent. Changing the model mid-project silently invalidates stored vectors. **Remedy:** delete `index.faiss` and `index_ids.json`, then rebuild from `memory.json` (original text remains in each item's `value` and `descriptor`), or run `scripts/clean.py` and re-index the corpus.

## Forward pointers

| Upgrade | What it addresses |
|---------|-------------------|
| **Semantic chunking** | Replaces sliding window; chunker as its own typed module |
| **Hybrid retrieval + RRF** | Dense baseline + sparse path; fusion inside `Memory.read`; same external interface |
| **Parallel fan-out (DAG agent)** | F run 1's five sequential `index_document` calls → concurrent nodes in `asyncio.TaskGroup` |
| **Skills** | Perception attaches capability labels (not tool names); Decision receives a filtered tool subset |
| **Cross-encoder reranking** | Second-stage ranker over hybrid top-*k* |
| **Knowledge-graph / entity consolidation** | Further extensions (mentioned only as pointers) |

## Corpus (`sandbox/papers/`)

Five markdown summaries (not full PDFs):

- `attention.md` — Transformer (self-attention, parallelism, positional encoding)
- `chain_of_thought.md` — linear stepwise reasoning, backprop through reasoning steps
- `react.md` — interleaved reasoning and tool actions
- `dpo.md` — preference optimisation, reward shaping
- `lora.md` — low-rank adaptation, distributed learning signal

The phrase **“credit assignment”** does not appear in any file (Query **G** relies on vector synonym recall).

## Architectural checkpoints

- **Tool-blindness**: `grep` over `cognitive_rag/perception.py` — no MCP tool names in the SYSTEM prompt.
- **Rendering**: `_format_hits` in `memory.py` exposes `value.text` / dates, not descriptor-only.
- **Cross-process memory**: FAISS reloaded from disk on every `memory.read()`.
