from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import get_settings
from .prompts import load_prompt
from .retrieval import PgVectorRetriever, RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Source: {chunk.title} | {chunk.source} | chunk {chunk.chunk_index}]\n{chunk.content}"
        for chunk in chunks
    )


def build_rag_chain(retriever: PgVectorRetriever, prompt_version: str):
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for answer generation; "
            "retrieval works without it."
        )

    prompt = ChatPromptTemplate.from_template(load_prompt(prompt_version))
    llm = ChatOpenAI(model=settings.llm_model, temperature=0)

    retrieve = RunnableLambda(
        lambda question: {"question": question, "chunks": retriever.search(question)}
    )
    answer = (
        RunnableLambda(
            lambda payload: {
                "question": payload["question"],
                "context": _format_context(payload["chunks"]),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return retrieve | RunnablePassthrough.assign(answer=answer)


def ask(retriever: PgVectorRetriever, question: str, prompt_version: str) -> dict:
    result = build_rag_chain(retriever, prompt_version).invoke(question)
    return {
        "question": question,
        "answer": result["answer"],
        "prompt_version": prompt_version,
        "sources": [chunk.as_dict() for chunk in result["chunks"]],
    }
