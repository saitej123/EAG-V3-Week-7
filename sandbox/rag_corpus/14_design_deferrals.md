# Design Deferrals (Forward Pointers)

These simplifications are deliberate. Each has a forward pointer to the upgrade that addresses it properly.

## Dense retrieval only

Vector retrieval in this codebase has no hybrid sparse partner. Production retrieval systems run a sparse retriever (BM25 or learned-sparse) and a dense retriever in parallel and combine the ranked lists with Reciprocal Rank Fusion (RRF). The current implementation uses dense retrieval alone via FAISS and gemini-embedding-2. Hybrid retrieval is planned for a future release.

## Heuristic chunking

Chunking is heuristic. The sliding window splits documents at arbitrary word boundaries (default 400 words, 80 overlap). Semantic chunking that respects sentences and section headers is deferred to a future release.

## FAISS reload on every read

The FAISS index is reloaded from disk on every memory.read() call so MCP subprocess index_document writes are visible to the agent. The cost is small at demo scale. At higher scale, memory-mapped indexes with file-modification-time invalidation or inter-process locks become appropriate. This codebase does not model that scale yet.

## Fixed embedding model

The gateway pins GEMINI_EMBED_MODEL (default gemini-embedding-2) so all vectors live in the same semantic space. Changing the model silently invalidates every vector previously stored. Remedy: delete index.faiss and index_ids.json, then rebuild from memory.json (original text remains in each descriptor and value), or run scripts/clean.py and re-index.

## Reciprocal Rank Fusion (concept only)

RRF merges ranked lists from multiple retrievers by summing 1/(k + rank) scores. It often outperforms score normalisation when sparse and dense retrievers emphasise different relevant documents. RRF is not implemented in memory.read yet — it is a forward pointer for hybrid retrieval in a future release.
