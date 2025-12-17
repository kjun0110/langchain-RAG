"""의존성 주입 - FastAPI 의존성."""

from functools import lru_cache
from ..service.rag_service import RAGService
from ..service.llm_service import LLMService
from ..service.embedding_service import EmbeddingService
from ..repository.vector_store import VectorStoreRepository
from ..config.settings import get_settings


@lru_cache()
def get_llm_service() -> LLMService:
    """LLM 서비스 의존성."""
    settings = get_settings()
    return LLMService(model_name=settings.default_llm_model)


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Embedding 서비스 의존성."""
    settings = get_settings()
    return EmbeddingService(embedding_name=settings.default_embedding_model)


@lru_cache()
def get_vector_store_repo() -> VectorStoreRepository:
    """벡터 스토어 리포지토리 의존성."""
    settings = get_settings()
    embedding_service = get_embedding_service()
    return VectorStoreRepository(
        connection_string=settings.connection_string,
        collection_name=settings.collection_name,
        embeddings=embedding_service.get_langchain_embeddings(),
    )


@lru_cache()
def get_rag_service() -> RAGService:
    """RAG 서비스 의존성."""
    llm_service = get_llm_service()
    embedding_service = get_embedding_service()
    vector_store_repo = get_vector_store_repo()
    return RAGService(
        vector_store_repo=vector_store_repo,
        llm_service=llm_service,
        embedding_service=embedding_service,
    )

