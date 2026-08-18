# Prompt Change Procedure

Prompts live in versioned text files in the repository. A prompt version is immutable after it has been used in a release; changes require a new version file. Evaluation runs log the prompt version even when the experiment measures retrieval-only metrics so the complete RAG configuration remains reproducible.
