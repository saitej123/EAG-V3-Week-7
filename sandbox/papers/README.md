# Papers (`sandbox/papers/`)

Eval corpus for queries **E–H** (single-doc index, bulk index, synonym recall, cross-paper synthesis).

Each summary is **~1,200+ words** (~3 index chunks at 400-word windows) distilled from the original publications and standard references.

| File | Topic | Chunks (approx.) |
|------|--------|------------------|
| `attention.md` | Transformer — self-attention, parallel layers, positional encoding | 3+ |
| `chain_of_thought.md` | Linear stepwise reasoning traces; gradient path through rationale tokens | 3+ |
| `react.md` | ReAct — interleaved thoughts and environment actions | 3+ |
| `dpo.md` | Direct Preference Optimization; pairwise log-prob margins | 3+ |
| `lora.md` | Low-rank adaptation; frozen base + trainable adapters | 3+ |

Optional PDF sidecars (`2605.23904v2.pdf`, `2605.20025v2.pdf`) may coexist for other demos; queries **E–H** target the `.md` summaries above.

Index via `index_document` (single file) or bulk: *Index every .md file under papers/* (Query F).
