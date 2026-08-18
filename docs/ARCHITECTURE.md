# Architecture

```text
                     +-------------------+
                     | Markdown / TXT    |
                     +---------+---------+
                               |
                               v
+---------+    enqueue    +----+-----+    split/embed    +------------------+
| CLI/API | ------------> |  Redis   | <--------------- | Celery worker    |
+----+----+               +----+-----+                  +--------+---------+
     |                          |                                 |
     | query                    | broker/result                   | vectors + metadata
     v                          v                                 v
+----+--------------------------+----+                    +-------+--------+
| LangChain runnable + retriever    | <----------------- | Postgres       |
| prompt version + optional LLM     |   cosine search    | + pgvector     |
+----------------+------------------+                    +----------------+
                 |
                 | retrieval experiments
                 v
          +------+-------+
          | MLflow       |
          | metrics +    |
          | registry     |
          +--------------+
```

## Design choices

- **Celery handles ingestion, not queries.** Embedding is the slow, batch-oriented step; keeping it off the API process makes the async boundary meaningful.
- **Postgres owns document metadata and vectors.** The project avoids a second vector database and demonstrates pgvector directly.
- **LangChain is intentionally thin.** It provides text splitting and the RAG runnable, while storage and evaluation logic remain ordinary Python/SQLAlchemy code.
- **Evaluation is retrieval-first.** Recall@k and MRR can be measured without paying for an LLM call, making experiments deterministic and inexpensive.
- **Prompt files are immutable versions.** The version is logged with every experiment so a run identifies the complete RAG configuration.
- **The MLflow registry entry is a configuration wrapper.** No custom embedding model is trained here. The registry stores the chosen retrieval configuration and its lineage under a `champion` alias.
