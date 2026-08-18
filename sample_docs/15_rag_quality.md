# Retrieval Quality Evaluation

RAG retrieval changes are evaluated on a labeled query set before release. The team records recall at k and mean reciprocal rank, along with chunk size, overlap, embedding model, and prompt version. A candidate configuration must not reduce recall at 5 by more than two percentage points relative to the current champion.
