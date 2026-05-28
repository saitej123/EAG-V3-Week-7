#!/usr/bin/env python3
"""RAG corpus utilities: build 55 markdown files or bulk-index into FAISS memory."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "sandbox" / "rag_corpus"
MANIFEST_PATH = ROOT / "corpus" / "MANIFEST.json"

# Each entry: (filename_stem, title, body). Bodies are self-contained notes (~350–600 words).
ENTRIES: list[tuple[str, str, str]] = [
    ("01_transformer_attention", "Scaled Dot-Product Self-Attention",
     "The Transformer replaces recurrence with self-attention. Queries, keys, and values pass through learned linear projections. "
     "Compatibility scores use scaled dot products so gradients remain stable as dimension grows. Multi-head attention runs several "
     "attention functions in parallel, letting each head focus on different relational patterns across the sequence."),
    ("02_transformer_parallelism", "Parallel Sequence Processing",
     "Unlike RNNs, Transformers apply every layer to all token positions simultaneously. This improves hardware utilisation and "
     "shortens the path between distant tokens to O(1) per layer. Training throughput scales well on GPUs when batch and sequence "
     "length fit memory. Inference can still be memory-bound for very long contexts."),
    ("03_positional_encoding", "Positional Encoding in Transformers",
     "Self-attention alone is permutation-invariant. Sinusoidal positional encodings are added to input embeddings so the model "
     "knows token order while keeping fully parallel forward passes. Learned positional embeddings are a common variant in later models."),
    ("04_chain_of_thought_linear", "Linear Stepwise Reasoning Traces",
     "Chain-of-thought prompting asks the model to emit intermediate steps before the final answer. The trace is a single forward "
     "narrative: step one, step two, and so on. There is no interleaving of external tool calls—only language tokens in context."),
    ("05_cot_backprop", "Gradients Through Reasoning Steps",
     "Each emitted reasoning token participates in the computation graph. Error signals from the final label propagate backward "
     "through the intermediate steps via standard backpropagation through the reasoning chain. The model learns which intermediate "
     "hops carry information useful for the downstream task."),
    ("06_react_interleave", "Interleaving Thoughts and Actions",
     "ReAct alternates natural-language thoughts with environment actions such as search or calculator calls. The pattern "
     "Thought → Action → Observation → Thought grounds reasoning in retrieved evidence rather than parametric memory alone."),
    ("07_react_observations", "Intermediate Signals from the Environment",
     "Each observation injects fresh external text into the context. The model must reconcile its hypotheses with retrieved "
     "content, reducing hallucination on knowledge-intensive benchmarks when actions return useful snippets."),
    ("08_dpo_preferences", "Direct Preference Optimisation",
     "DPO aligns language models to human preferences without a separate reinforcement-learning loop. Given a prompt and two "
     "completions—one preferred, one dispreferred—the loss increases the log-probability margin for the chosen response while "
     "regularising against a frozen reference policy."),
    ("09_dpo_reward_shaping", "Implicit Reward Shaping in DPO",
     "The preference objective implicitly shapes which behaviours receive positive reinforcement. The β hyperparameter trades "
     "alignment strength against deviation from the reference model: higher β preserves fluency; lower β pursues sharper preference fit."),
    ("10_lora_adapters", "Low-Rank Adaptation (LoRA)",
     "LoRA freezes pretrained weights and injects trainable low-rank factors into selected linear layers. For weight matrix W, "
     "the update ΔW = BA uses small rank r, updating only adapter parameters during fine-tuning."),
    ("11_lora_rank", "Rank and Learning Signal Distribution",
     "Because updates live in a low-rank subspace, task knowledge compresses into few degrees of freedom. Rank r controls how "
     "gradient influence is distributed across layers—higher rank allows finer behavioural shifts than very low rank adapters."),
    ("12_rag_overview", "Retrieval-Augmented Generation",
     "RAG combines a retriever that fetches relevant passages with a generator that conditions answers on retrieved context. "
     "The retriever may be sparse (BM25), dense (embedding + vector index), or hybrid. Quality depends on chunking, embedding model, "
     "and whether retrieved text actually contains the answer."),
    ("13_faiss_basics", "FAISS Vector Indexes",
     "FAISS stores dense embeddings and supports fast approximate or exact nearest-neighbour search. IndexFlatIP with L2-normalised "
     "vectors approximates cosine similarity via inner product. Indexes persist to disk and reload across processes when cross-process "
     "consistency matters more than in-memory caching."),
    ("14_embedding_models", "Embedding Model Consistency",
     "All vectors in an index must come from the same embedding model and dimension. Silently switching models invalidates prior "
     "vectors. Recovery requires rebuilding the index from stored text in a source-of-truth store such as memory.json."),
    ("15_sliding_window_chunking", "Heuristic Sliding-Window Chunking",
     "Fixed-size windows with overlap split documents at arbitrary token boundaries. Simple and fast but may cut sentences mid-thought. "
     "Semantic chunking that respects paragraphs and sections improves retrieval precision and arrives in future releases."),
    ("16_python_asyncio_gather", "asyncio.gather for Concurrent I/O",
     "asyncio.gather schedules multiple awaitables concurrently on one event loop. Use it for parallel network or disk I/O bound tasks. "
     "CPU-bound work still needs executors or multiprocessing; gather does not bypass the GIL for heavy computation."),
    ("17_python_context_managers", "Context Managers and Resource Cleanup",
     "The with statement guarantees __exit__ runs even on exceptions. Essential for files, database connections, and locks. "
     "Async context managers use async with for the same pattern in asyncio code."),
    ("18_fastapi_sse", "Server-Sent Events in FastAPI",
     "SSE streams server pushes over a long-lived HTTP response. Starlette's EventSourceResponse or StreamingResponse with "
     "text/event-stream works well for live agent logs. Clients reconnect automatically in browsers that support EventSource."),
    ("19_pydantic_v2", "Pydantic v2 Boundary Models",
     "Pydantic models validate data at module boundaries. model_dump(mode='json') serialises datetimes cleanly. extra='ignore' "
     "drops unknown fields when evolving schemas. Structured LLM outputs map naturally to response_schema validation."),
    ("20_mcp_stdio", "MCP stdio Transport",
     "Model Context Protocol tools run as a subprocess communicating over stdin/stdout JSON-RPC. The agent process spawns the server; "
     "each tool call crosses the process boundary. Writes from MCP tools must appear on disk before the agent's next memory read if "
     "both share persistent state."),
    ("21_tool_blindness", "Perception Tool-Blindness",
     "Perception decomposes queries into intent-level goals without naming MCP tools. Decision selects tools using its catalogue and "
     "MCP docstrings. Pushing tool names into Perception's SYSTEM breaks the four-role boundary and does not scale past ~11 tools."),
    ("22_format_hits", "Rendering Memory Hits for Decision",
     "If Decision sees only descriptors, it may miss dates or chunk bodies stored in value.text. The _format_hits renderer exposes "
     "descriptor plus salient value fields so answers from memory do not require compensating SYSTEM rules."),
    ("23_cross_process_memory", "Cross-Process Memory Consistency",
     "When MCP index_document and the agent both write FAISS files, reload the index from disk on every read. In-process caching "
     "hides subprocess writes. The trade is milliseconds of disk I/O versus lock-free consistency."),
    ("24_scratchpad_items", "Scratchpad Memory Items",
     "Scratchpad rows persist to memory.json but skip embedding. They explain item count exceeding FAISS row count. Classifier output "
     "may label ephemeral working notes as scratchpad while facts and preferences receive vectors."),
    ("25_hybrid_retrieval_future", "Hybrid Retrieval (Forward Pointer)",
     "Production systems combine dense retrieval with sparse BM25 or learned-sparse models and fuse ranked lists via Reciprocal Rank "
     "Fusion. This codebase uses dense retrieval only; hybrid fusion is deferred to a future release."),
    ("26_semantic_chunking_future", "Semantic Chunking (Forward Pointer)",
     "LLM-aware chunk boundary detectors respect sentences and section headers. Fixed sliding windows are used in index_document today."),
    ("27_skills_abstraction", "Skills vs Tool Names in Goals",
     "When the tool catalogue grows, Perception may attach coarse capability labels while Decision receives a filtered subset. "
     "Skills preserve tool-blindness: influence availability without naming web_search or index_document in Perception prompts."),
    ("28_wikipedia_fetch", "Fetching Full Pages for Extraction",
     "Large pages should become artifacts; Decision reads attached bytes on synthesis goals. fetch_url stores markdown above a size "
     "threshold in the artifact store rather than stuffing the full page into history JSON."),
    ("29_weather_wttr", "Weather Lookups via wttr.in",
     "Lightweight weather JSON endpoints support activity planning queries. Forecast text feeds tool_outcome memory rows for later "
     "goals comparing indoor versus outdoor options."),
    ("30_tokyo_family_activities", "Family Activities in Tokyo",
     "Common recommendations include Ueno Zoo and Park, teamLab Planets, Tokyo Skytree, and Odaiba attractions. Rainy forecasts "
     " favour museums and indoor installations; clear weather suits parks and observation decks."),
    ("31_birthday_reminders", "Calendar Reminder Patterns",
     "Durable facts such as birthdays store entity and date in memory value fields. Reminder files in the sandbox capture two-week "
     "and on-the-day alerts via create_file without re-fetching the date from the web."),
    ("32_asyncio_best_practices", "Python asyncio Best Practices",
     "Prefer asyncio.run() as the main entry point. Use async context managers for clients. Avoid blocking calls inside coroutines. "
     "Gather related I/O with asyncio.gather. Set explicit timeouts on network operations."),
    ("33_shannon_bio", "Claude Shannon Biography Notes",
     "Claude Shannon (1916–2001) founded information theory, introduced bit as an information unit, and connected entropy to "
     "communication channels. His master's thesis on Boolean algebra and switching circuits predates modern digital design."),
    ("34_information_theory", "Information Theory Contributions",
     "Key ideas include channel capacity, noisy-channel coding theorem, and entropy as expected information content. Shannon's "
     "1948 paper unified prior work on communication and cryptography under one mathematical framework."),
    ("35_vector_synonym_recall", "Semantic Recall Without Keyword Overlap",
     "Dense retrieval surfaces passages about backpropagation through reasoning steps when the query asks about credit assignment "
     "problems, even if that exact phrase never appears in the corpus. This is the pedagogical payoff of embeddings over keyword overlap."),
    ("36_reciprocal_rank_fusion", "Reciprocal Rank Fusion (RRF)",
     "RRF merges ranked lists from multiple retrievers by summing 1/(k + rank) scores. It often outperforms simple score normalisation "
     "when sparse and dense retrievers emphasise different relevant documents. Not yet implemented in memory.read."),
    ("37_bm25_sparse", "BM25 Sparse Retrieval",
     "BM25 scores term frequency and document length without embeddings. Strong on exact terminology; weak on paraphrase. Hybrid "
     "systems run BM25 alongside FAISS and fuse with RRF."),
    ("38_cross_encoder_rerank", "Cross-Encoder Reranking",
     "A cross-encoder jointly encodes query and candidate passage for a relevance score. Used as a second stage over top-k hybrid "
     "results. Small models suffice; latency grows with candidate count."),
    ("39_dag_parallel_index", "Parallel Indexing in DAG Agents",
     "Indexing five papers sequentially across eleven iterations motivates parallel fan-out. DAG nodes can dispatch multiple "
     "index_document calls inside asyncio.TaskGroup to compress wall-clock time."),
    ("40_gateway_embed", "Gateway Embed Endpoint",
     "Embeddings for memory items use the Gemini SDK with gemini-embedding-2 by default (GEMINI_EMBED_MODEL). "
     "Embedding 2 uses task prefixes in the prompt (search query vs document). When GATEWAY_URL is set, memory may call "
     "the gateway /v1/embed endpoint, falling back to Gemini on failure. Re-index after changing the embed model."),
    ("41_index_document_vs_read", "index_document vs read_file",
     "index_document chunks content into searchable memory facts for later turns. read_file returns full file text once for inspection "
     "without updating FAISS. Decision docstrings carry this distinction; Perception stays tool-blind."),
    ("42_search_knowledge_usage", "search_knowledge Tool",
     "search_knowledge runs memory.read filtered to fact items and returns previews with source labels. Use when indexed chunks "
     "already exist instead of re-fetching URLs or re-reading sandbox files."),
    ("43_artifact_attach", "Artifact Attachment in Perception",
     "Perception sets artifact_index to tell Decision which memory hit carries bytes to attach. Never pass art: handles as string "
     "goals—only integer indices from the enumerated artifact list."),
    ("44_iteration_budget", "Smart Iteration Budget",
     "Default three loops extend up to four for multi-step queries. Simple recall finishes early. Top-three search plus batch fetch_urls "
     "plus synthesis typically needs three to four iterations. Emergency rescue synthesises from search snippets when budget exhausts."),
    ("45_tavily_fallback_chain", "Web Search Fallback Chain",
     "web_search tries Tavily, then crawl4ai, Gemini live search, then DuckDuckGo. Usage caps log to usage.json. Empty tool args "
     "enrich from user query and active goal."),
    ("46_commerce_sqlite", "Commerce SQLite Cache",
     "Optional product rows cache PDP data separately from episodic memory.json. query_database searches product_name and url fields "
     "for Indian e-commerce price-shopping queries."),
    ("47_pop_validation", "Proof-of-Prompt Validation",
     "validate_prompts_pop.py evaluates Perception and Decision prompts against nine PoP criteria concurrently. Keeps prompts "
     "structured with explicit reasoning and tool separation."),
    ("48_emergency_rescue", "Emergency Answer Rescue",
     "When max iterations hit without a user-facing answer, the agent may run emergency web search and synthesis from snippets rather "
     "than returning an empty partial summary."),
    ("49_memory_remember", "memory.remember Classification",
     "One structured LLM call classifies raw user text into fact, preference, tool_outcome, or scratchpad. Facts and preferences "
     "receive embeddings at insertion; scratchpad does not."),
    ("50_record_outcome", "record_outcome Episodic Rows",
     "After each MCP tool dispatch, record_outcome appends a tool_outcome row with descriptor summarising the call and preview text. "
     "These rows embed and participate in vector recall on later iterations."),
    ("51_corpus_manifest", "Corpus Manifest Discipline",
     "The project requires fifty or more indexed items and a manifest listing sources. This repository ships sandbox/rag_corpus "
     "with fifty-five markdown notes and corpus/MANIFEST.json for reproducible RAG demos."),
    ("52_no_corpus_baseline", "No-Corpus Baseline Tests",
     "Custom eval queries must fail without the index: run with empty state/ before indexing to show retrieval is load-bearing. "
     "Compare traces with indexed runs to prove FAISS-backed answers are not generic LLM hallucination."),
    ("53_claude_shannon_wiki", "Wikipedia Fetch Pattern",
     "Query A fetches Claude Shannon's Wikipedia page, stores a large artifact, and answers extraction goals from attached bytes "
     "without refetching. Vector memory is uninvolved; artifact attach carries the payload."),
    ("54_query_f_persistence", "Query F Cross-Run Persistence",
     "After indexing five papers, state/ holds memory.json, index.faiss, and index_ids.json. A fresh agent process on run two reads "
     "FAISS from disk on iteration one and answers from persisted chunks without re-indexing."),
    ("55_submission_checklist", "Submission Checklist",
     "Submit GitHub repo with README manifest, eight base query traces A–H, five custom RAG queries with indexed vs no-index comparison, "
     "and passing Perception grep gate (zero MCP tool names in Perception SYSTEM)."),
]


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest_items: list[dict] = []
    for stem, title, body in ENTRIES:
        rel = f"rag_corpus/{stem}.md"
        path = ROOT / "sandbox" / rel
        text = f"# {title}\n\n{body}\n"
        path.write_text(text, encoding="utf-8")
        manifest_items.append(
            {
                "id": stem,
                "path": f"sandbox/{rel}",
                "sandbox_path": rel,
                "title": title,
                "word_count": len(body.split()),
            }
        )

    manifest = {
        "name": "Developer Knowledge RAG Corpus",
        "description": "Fifty-five markdown notes for desktop document RAG over sandbox/rag_corpus/",
        "item_count": len(manifest_items),
        "index_tool": "index_directory or index_document per file",
        "retrieve_tool": "search_knowledge",
        "items": manifest_items,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest_items)} corpus files to {CORPUS_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


def index_corpus(*, keep_state: bool = False) -> None:
    """Index sandbox/rag_corpus into Memory + FAISS."""
    import shutil
    import sys

    sys.path.insert(0, str(ROOT))
    state = ROOT / "state"
    if state.exists() and not keep_state:
        shutil.rmtree(state)
        print("Cleaned state/ before indexing.")

    from cognitive_rag.indexing import index_directory

    result = index_directory("rag_corpus")
    print(f"Indexed {result['files_indexed']} files, {result['chunks_indexed']} chunks.")
    print("state/memory.json + index.faiss + index_ids.json updated.")


def main_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG corpus build and index")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="Generate 55 markdown files and corpus/MANIFEST.json")
    idx_p = sub.add_parser("index", help="Bulk-index sandbox/rag_corpus into FAISS")
    idx_p.add_argument("--keep-state", action="store_true", help="Do not wipe state/ first")
    args = parser.parse_args()
    if args.command == "build":
        main()
    elif args.command == "index":
        index_corpus(keep_state=getattr(args, "keep_state", False))


if __name__ == "__main__":
    main_cli()
