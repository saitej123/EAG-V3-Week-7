# FAISS Vector Indexes

FAISS stores dense embeddings and supports fast approximate or exact nearest-neighbour search. IndexFlatIP with L2-normalised vectors approximates cosine similarity via inner product. Indexes persist to disk and reload across processes when cross-process consistency matters more than in-memory caching.
