"""Embedding 서비스 - Embedding 모델 사용."""

from typing import List, Optional, Dict, Any
from ..model.base import BaseEmbedding
from ..model.factory import EmbeddingFactory


class EmbeddingService:
    """Embedding 서비스."""

    def __init__(self, embedding_name: Optional[str] = None):
        """Embedding 서비스 초기화."""
        self.embedding: BaseEmbedding = EmbeddingFactory.get(embedding_name)

    def embed_query(self, text: str) -> List[float]:
        """단일 텍스트를 벡터로 변환."""
        return self.embedding.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 벡터로 변환."""
        return self.embedding.embed_documents(texts)

    def get_langchain_embeddings(self):
        """LangChain 호환 Embeddings 반환."""
        return self.embedding.get_langchain_embeddings()

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환."""
        return self.embedding.get_model_info()

