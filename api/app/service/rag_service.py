"""RAG 서비스 - Retrieval-Augmented Generation."""

from typing import List, Optional, Dict, Any
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .llm_service import LLMService
from .embedding_service import EmbeddingService
from ..repository.vector_store import VectorStoreRepository


class RAGService:
    """RAG 서비스."""

    def __init__(
        self,
        vector_store_repo: VectorStoreRepository,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
    ):
        """RAG 서비스 초기화."""
        self.vector_store_repo = vector_store_repo
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.rag_chain: Optional[Runnable] = None
        self._initialize_rag_chain()

    def _initialize_rag_chain(self) -> None:
        """RAG 체인 초기화."""
        # Retriever 생성
        retriever = self.vector_store_repo.get_retriever(search_kwargs={"k": 3})

        # 대화 기록을 고려한 검색 쿼리 생성 프롬프트
        contextualize_q_system_prompt = (
            "대화 기록과 최신 사용자 질문이 주어졌을 때, "
            "대화 기록의 맥락을 참고하여 독립적으로 이해할 수 있는 질문으로 재구성하세요. "
            "질문에 답하지 말고, 필요시 재구성하고 그렇지 않으면 그대로 반환하세요."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 대화 기록을 고려한 Retriever 생성
        llm = self.llm_service.get_langchain_model()
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # 질문 답변 프롬프트
        qa_system_prompt = (
            "당신은 LangChain과 PGVector를 사용하는 도움이 되는 AI 어시스턴트입니다. "
            "다음 검색된 컨텍스트 정보를 참고하여 사용자의 질문에 답변해주세요. "
            "컨텍스트에 답변할 수 없는 질문이면, 정중하게 그렇게 말씀해주세요. "
            "답변은 최대 3문장으로 간결하게 작성해주세요.\n\n"
            "컨텍스트:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 문서 결합 체인 생성
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # 최종 RAG 체인 생성
        self.rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

    def generate(
        self, query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """RAG를 사용한 텍스트 생성."""
        if not self.rag_chain:
            raise RuntimeError("RAG 체인이 초기화되지 않았습니다.")

        # 대화 기록을 LangChain 메시지 형식으로 변환
        chat_history = []
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    chat_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    chat_history.append(AIMessage(content=msg.get("content", "")))

        # RAG 체인 실행
        result = self.rag_chain.invoke(
            {
                "input": query,
                "chat_history": chat_history,
            }
        )

        # 체인 결과에서 답변 추출
        return result.get("answer", "답변을 생성할 수 없습니다.")

