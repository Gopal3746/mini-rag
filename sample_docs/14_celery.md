# Background Job Operations

Embedding and document-processing work runs in Celery workers through Redis. Jobs should be idempotent because a worker may retry after a process failure. The default retry policy uses exponential backoff for transient network failures. Large payloads belong in object storage or the database; queue messages should carry identifiers rather than full documents.
