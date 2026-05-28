# VLM Document Indexing Pipeline

This project indexes documents through a unified vision pipeline: any supported file is normalized to PDF, each page is rasterized, and Gemini vision extracts structured text with page-level citations.

Supported inputs include PDF, images, markdown, plain text, and Office formats when LibreOffice is available. Each indexed page stores `page_number`, `page_total`, and a citation string such as `papers/foo.pdf p.3/12` in FAISS memory facts.

The VLM prompt captures checkboxes, tables, and figure summaries—not only raw OCR text—so retrieval can ground answers with page references. Use `index_document` for single files and `index_directory` for bulk ingestion under `sandbox/rag_corpus/` or `sandbox/papers/`.
