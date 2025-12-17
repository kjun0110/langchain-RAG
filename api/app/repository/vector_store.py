"""벡터 스토어 리포지토리."""

from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


class VectorStoreRepository:
    """벡터 스토어 리포지토리."""

    def __init__(
        self,
        connection_string: str,
        collection_name: str,
        embeddings: Embeddings,
    ):
        """벡터 스토어 리포지토리 초기화."""
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.vector_store: Optional[PGVector] = None
        self._initialize()

    def _initialize(self) -> None:
        """벡터 스토어 초기화."""
        try:
            # 기존 컬렉션이 있으면 로드
            self.vector_store = PGVector(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_string=self.connection_string,
            )
        except Exception:
            # 컬렉션이 없으면 생성 (초기 문서로)
            self.vector_store = PGVector.from_documents(
                embedding=self.embeddings,
                documents=[
                    Document(
                        page_content="LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.",
                        metadata={"source": "intro"},
                    ),
                    Document(
                        page_content="pgvector는 PostgreSQL에서 벡터 검색을 가능하게 하는 확장입니다.",
                        metadata={"source": "pgvector"},
                    ),
                    Document(
                        page_content="Hello World는 프로그래밍의 첫 번째 예제입니다.",
                        metadata={"source": "hello"},
                    ),
                ],
                collection_name=self.collection_name,
                connection_string=self.connection_string,
            )

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> BaseRetriever:
        """Retriever 반환."""
        if not self.vector_store:
            raise RuntimeError("벡터 스토어가 초기화되지 않았습니다.")
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs or {"k": 3}
        )

    def similarity_search(
        self, query: str, k: int = 3
    ) -> List[Document]:
        """유사도 검색."""
        if not self.vector_store:
            raise RuntimeError("벡터 스토어가 초기화되지 않았습니다.")
        return self.vector_store.similarity_search(query, k=k)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """문서 추가."""
        if not self.vector_store:
            raise RuntimeError("벡터 스토어가 초기화되지 않았습니다.")
        return self.vector_store.add_documents(documents)

