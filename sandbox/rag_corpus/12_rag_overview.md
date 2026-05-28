# Retrieval-Augmented Generation

RAG combines a retriever that fetches relevant passages with a generator that conditions answers on retrieved context. The retriever may be sparse (BM25), dense (embedding + vector index), or hybrid. Quality depends on chunking, embedding model, and whether retrieved text actually contains the answer.
