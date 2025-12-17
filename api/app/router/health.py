"""헬스 체크 API 라우터."""

from fastapi import APIRouter
from ..model.factory import LLMFactory, EmbeddingFactory

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """헬스 체크 엔드포인트."""
    return {
        "status": "healthy",
        "llm_models": LLMFactory.list_models(),
        "default_llm": LLMFactory.get_default(),
        "embedding_models": EmbeddingFactory.list_embeddings(),
        "default_embedding": EmbeddingFactory.get_default(),
    }

