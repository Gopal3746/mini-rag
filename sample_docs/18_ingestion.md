# Document Ingestion Contract

The ingestion service accepts UTF-8 Markdown and text files from an approved mounted directory. Ingestion computes a SHA-256 checksum per document, chunks the content, creates embeddings, and stores document and chunk metadata in PostgreSQL. Re-ingesting the same configuration replaces its chunks so repeated jobs are idempotent.
