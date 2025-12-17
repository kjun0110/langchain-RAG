"""
FastAPI 백엔드 서버 - FastAPI와 LangChain 연동.

이 서버는 FastAPI를 통해 LangChain RAG 체인을 제공하는 API 서버입니다.
worker 서비스(app.py)가 초기화한 pgvector 벡터 스토어를 사용하여
챗봇 API를 제공합니다.

역할:
- FastAPI 서버 제공 (REST API)
- LangChain RAG 체인 실행
- worker가 초기화한 pgvector 벡터 스토어 활용
"""

import os
import time
import warnings
from pathlib import Path
from typing import List, Optional

# .env 파일 로드 (프로젝트 루트에서 찾기)
try:
    from dotenv import load_dotenv

    # 프로젝트 루트 찾기 (api/app/ -> api/ -> 프로젝트 루트)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
    else:
        # 현재 디렉토리에서도 시도
        load_dotenv()
except ImportError:
    pass  # python-dotenv가 없으면 환경 변수만 사용

# PGVector의 JSONB deprecation 경고 무시
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings(
        "ignore",
        category=LangChainPendingDeprecationWarning,
        module="langchain_community.vectorstores.pgvector",
    )
except ImportError:
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="langchain_community.vectorstores.pgvector",
    )

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="langchain_community.vectorstores.pgvector",
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

# Neon PostgreSQL 연결 문자열 (.env 파일의 DATABASE_URL 사용)
DATABASE_URL = os.getenv("DATABASE_URL")
SSLMODE = os.getenv("sslmode", "require")

if DATABASE_URL:
    # DATABASE_URL에 sslmode가 없으면 추가
    if "sslmode=" not in DATABASE_URL:
        separator = "&" if "?" in DATABASE_URL else "?"
        CONNECTION_STRING = f"{DATABASE_URL}{separator}sslmode={SSLMODE}"
    else:
        CONNECTION_STRING = DATABASE_URL
else:
    # 기본값 (fallback)
    CONNECTION_STRING = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql://neondb_owner:npg_bNXv7Ll1mrBJ@ep-empty-tree-a15rzl4v-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
    )

COLLECTION_NAME = "langchain_collection"

# FastAPI 앱 생성
app = FastAPI(
    title="LangChain Chatbot API",
    description="PGVector와 연동된 LangChain 챗봇 API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
vector_store: Optional[PGVector] = None
openai_embeddings = None
local_embeddings = None
openai_llm = None
local_llm = None
openai_rag_chain: Optional[Runnable] = None
local_rag_chain: Optional[Runnable] = None
# 할당량 초과 추적
openai_quota_exceeded = False


def wait_for_postgres(max_retries: int = 30, delay: int = 2) -> None:
    """Neon PostgreSQL이 준비될 때까지 대기."""
    import psycopg2

    print(
        f"[INFO] Neon PostgreSQL 연결 시도 중... (연결 문자열: {CONNECTION_STRING[:50]}...)"
    )

    for i in range(max_retries):
        try:
            conn = psycopg2.connect(CONNECTION_STRING)

            # PGVector 확장 확인
            cur = conn.cursor()
            cur.execute(
                "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
            )
            vector_ext = cur.fetchone()

            if vector_ext:
                print("[OK] Neon PostgreSQL 연결 성공!")
                print(f"[INFO] PGVector 확장 설치됨 (버전: {vector_ext[1]})")
            else:
                print("[OK] Neon PostgreSQL 연결 성공!")
                print("[WARNING] PGVector 확장이 설치되지 않았습니다!")

            conn.close()
            return
        except Exception as e:
            if i < max_retries - 1:
                print(
                    f"[INFO] Neon PostgreSQL 대기 중... ({i + 1}/{max_retries}) - {str(e)[:100]}"
                )
                time.sleep(delay)
            else:
                raise ConnectionError(f"Neon PostgreSQL 연결 실패: {e}")


def initialize_embeddings():
    """Embedding 모델 초기화 - OpenAI와 로컬 모델 모두 초기화."""
    global openai_embeddings, local_embeddings, openai_quota_exceeded
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # OpenAI Embedding 초기화
    if openai_api_key and openai_api_key != "your-api-key-here":
        try:
            openai_embeddings = OpenAIEmbeddings()
            # 간단한 테스트
            openai_embeddings.embed_query("test")
            print("[OK] OpenAI Embedding 모델 초기화 완료")
        except Exception as e:
            error_msg = str(e)
            if (
                "quota" in error_msg.lower()
                or "429" in error_msg
                or "insufficient_quota" in error_msg
            ):
                openai_quota_exceeded = True
                print(f"[WARNING] OpenAI API 할당량 초과: {error_msg[:100]}...")
                print("   OpenAI Embedding을 사용할 수 없습니다.")
                openai_embeddings = None
            else:
                print(f"[WARNING] OpenAI Embedding 초기화 실패: {error_msg[:100]}...")
                openai_embeddings = None
    else:
        print("[WARNING] OpenAI API 키가 설정되지 않았습니다.")
        openai_embeddings = None

    # 로컬 Embedding 초기화
    try:
        # langchain-huggingface 사용 (deprecation 경고 해결)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            # fallback to langchain_community
            from langchain_community.embeddings import HuggingFaceEmbeddings

        local_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # 간단한 테스트
        local_embeddings.embed_query("test")
        print("[OK] 로컬 Embedding 모델 초기화 완료 (sentence-transformers)")
    except Exception as local_error:
        print(f"[WARNING] 로컬 Embedding 모델 초기화 실패: {str(local_error)[:100]}...")
        local_embeddings = None

    if not openai_embeddings and not local_embeddings:
        raise RuntimeError(
            "OpenAI와 로컬 Embedding 모델 모두 초기화에 실패했습니다. "
            "OpenAI API 키를 설정하거나 sentence-transformers를 설치해주세요."
        )


def initialize_llm():
    """LLM 모델 초기화 - OpenAI와 로컬 모델 모두 초기화."""
    global openai_llm, local_llm, openai_quota_exceeded
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # OpenAI LLM 초기화
    if openai_api_key and openai_api_key != "your-api-key-here":
        try:
            openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            # 실제 API 호출 테스트 (할당량 확인)
            openai_llm.invoke("test")
            print("[OK] OpenAI Chat 모델 초기화 완료")
        except Exception as e:
            error_msg = str(e)
            if (
                "quota" in error_msg.lower()
                or "429" in error_msg
                or "insufficient_quota" in error_msg
            ):
                openai_quota_exceeded = True
                print(f"[WARNING] OpenAI API 할당량 초과: {error_msg[:100]}...")
                print("   OpenAI LLM을 사용할 수 없습니다.")
                openai_llm = None
            else:
                print(f"[WARNING] OpenAI Chat 모델 초기화 실패: {error_msg[:100]}...")
                openai_llm = None
    else:
        print("[WARNING] OpenAI API 키가 설정되지 않았습니다.")
        openai_llm = None

    # 로컬 Midm LLM 초기화
    try:
        from app.model.midm import load_midm_model

        # .env 파일에서 LOCAL_MODEL_DIR 읽기
        local_model_dir = os.getenv("LOCAL_MODEL_DIR")
        if local_model_dir:
            # 상대 경로를 절대 경로로 변환
            from pathlib import Path

            if not Path(local_model_dir).is_absolute():
                # 프로젝트 루트 기준으로 변환
                project_root = Path(__file__).parent.parent.parent
                local_model_dir = str(project_root / local_model_dir)
            print(f"[INFO] 로컬 모델 디렉토리: {local_model_dir}")
            midm_model = load_midm_model(
                model_path=local_model_dir, register=False, is_default=False
            )
        else:
            midm_model = load_midm_model(register=False, is_default=False)

        local_llm = midm_model.get_langchain_model()
        print("[OK] 로컬 Midm LLM 모델 초기화 완료")
    except Exception as local_error:
        error_msg = str(local_error)
        print(f"[WARNING] 로컬 Midm 모델 초기화 실패: {error_msg[:200]}...")
        import traceback

        print(f"[DEBUG] 상세 오류: {traceback.format_exc()[:500]}")
        local_llm = None

    if not openai_llm and not local_llm:
        raise RuntimeError(
            "OpenAI와 로컬 LLM 모델 모두 초기화에 실패했습니다. "
            "OpenAI API 키를 설정하거나 Midm 모델을 확인해주세요."
        )


def initialize_vector_store():
    """PGVector 스토어 초기화."""
    global vector_store, openai_embeddings, local_embeddings

    # LLM_PROVIDER에 따라 적절한 embedding 선택
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # 사용할 embedding 모델 선택 (우선순위: LLM_PROVIDER에 맞는 모델 > OpenAI > 로컬)
    if llm_provider == "midm" and local_embeddings:
        current_embeddings = local_embeddings
        print("[INFO] 로컬 Embedding 모델 사용 (LLM_PROVIDER=midm)")
    elif openai_embeddings:
        current_embeddings = openai_embeddings
        print("[INFO] OpenAI Embedding 모델 사용")
    elif local_embeddings:
        current_embeddings = local_embeddings
        print("[INFO] 로컬 Embedding 모델 사용 (fallback)")
    else:
        raise RuntimeError("사용 가능한 Embedding 모델이 없습니다.")

    try:
        print("[INFO] ===== PGVector 연결 확인 시작 =====")
        print(f"[INFO] 컬렉션 이름: {COLLECTION_NAME}")
        print(f"[INFO] 연결 문자열: {CONNECTION_STRING[:60]}...")

        # 기존 컬렉션이 있고 벡터 데이터가 있는지 확인
        try:
            print("[INFO] PGVector 객체 생성 중...")
            vector_store = PGVector(
                embedding_function=current_embeddings,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )
            print("[OK] PGVector 객체 생성 완료")

            # 벡터 데이터가 있는지 확인
            import psycopg2

            print("[INFO] 데이터베이스에서 벡터 데이터 확인 중...")
            conn = psycopg2.connect(CONNECTION_STRING)
            cur = conn.cursor()

            # 컬렉션 UUID 확인
            cur.execute(
                f"""
                SELECT uuid FROM langchain_pg_collection WHERE name = '{COLLECTION_NAME}'
            """
            )
            collection_result = cur.fetchone()

            if collection_result:
                collection_uuid = collection_result[0]
                print(f"[INFO] 컬렉션 UUID: {collection_uuid}")

                # 벡터 개수 확인
                cur.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM langchain_pg_embedding
                    WHERE collection_id = '{collection_uuid}'
                """
                )
                vector_count = cur.fetchone()[0]

                # 벡터 차원 확인
                cur.execute(
                    f"""
                    SELECT array_length(embedding::vector, 1) as vector_dim
                    FROM langchain_pg_embedding
                    WHERE collection_id = '{collection_uuid}'
                    LIMIT 1
                """
                )
                dim_result = cur.fetchone()
                vector_dim = dim_result[0] if dim_result and dim_result[0] else None

                conn.close()

                if vector_count > 0:
                    print("[OK] 기존 PGVector 스토어 로드 완료")
                    print(f"[INFO] 벡터 데이터 개수: {vector_count}개")
                    if vector_dim:
                        print(f"[INFO] 벡터 차원: {vector_dim}차원")
                    print("[OK] ===== PGVector 연결 확인 완료 =====")
                else:
                    # 컬렉션은 있지만 벡터 데이터가 없으면 초기 문서 추가
                    print("[INFO] 컬렉션은 존재하지만 벡터 데이터가 없습니다.")

                    # 기존 문서 확인 (중복 방지)
                    cur.execute(
                        f"""
                        SELECT DISTINCT document
                        FROM langchain_pg_embedding
                        WHERE collection_id = '{collection_uuid}'
                    """
                    )
                    existing_docs = {row[0] for row in cur.fetchall()}

                    initial_docs = [
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
                    ]

                    # 중복되지 않은 문서만 추가
                    docs_to_add = [
                        doc
                        for doc in initial_docs
                        if doc.page_content not in existing_docs
                    ]

                    if docs_to_add:
                        print(f"[INFO] 초기 문서 {len(docs_to_add)}개를 추가합니다...")
                        vector_store.add_documents(docs_to_add)
                        print("[OK] 초기 문서 추가 완료")
                    else:
                        print(
                            "[INFO] 초기 문서가 이미 모두 존재합니다. 추가하지 않습니다."
                        )

                    # 추가 후 확인
                    conn = psycopg2.connect(CONNECTION_STRING)
                    cur = conn.cursor()
                    cur.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM langchain_pg_embedding
                        WHERE collection_id = '{collection_uuid}'
                    """
                    )
                    final_count = cur.fetchone()[0]

                    # 고유 문서 개수 확인
                    cur.execute(
                        f"""
                        SELECT COUNT(DISTINCT document)
                        FROM langchain_pg_embedding
                        WHERE collection_id = '{collection_uuid}'
                    """
                    )
                    unique_doc_count = cur.fetchone()[0]
                    conn.close()
                    print(f"[INFO] 현재 벡터 데이터 개수: {final_count}개")
                    print(f"[INFO] 고유 문서 개수: {unique_doc_count}개")
                    if final_count > unique_doc_count:
                        print(
                            f"[WARNING] 중복된 벡터가 {final_count - unique_doc_count}개 있습니다."
                        )
                    print("[OK] ===== PGVector 연결 확인 완료 =====")
            else:
                conn.close()
                print("[WARNING] 컬렉션이 데이터베이스에 존재하지 않습니다.")

        except Exception as e:
            # 컬렉션이 없으면 초기 문서로 생성
            error_msg = str(e)
            print("[INFO] 컬렉션 로드 실패, 새로 생성합니다...")
            print(f"[INFO] 오류 내용: {error_msg[:150]}")
            print("[INFO] 초기 문서로 PGVector 스토어 생성 중...")
            vector_store = PGVector.from_documents(
                embedding=current_embeddings,
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
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )
            print("[OK] PGVector 스토어 생성 완료")

            # 생성 후 벡터 개수 확인
            import psycopg2

            conn = psycopg2.connect(CONNECTION_STRING)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = '{COLLECTION_NAME}'
                )
            """
            )
            vector_count = cur.fetchone()[0]
            conn.close()
            print(f"[INFO] 생성된 벡터 데이터 개수: {vector_count}개")
            print("[OK] ===== PGVector 연결 확인 완료 =====")
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] PGVector 스토어 초기화 실패: {error_msg[:200]}...")
        raise


def create_rag_chain(llm_model, embeddings_model):
    """RAG 체인 생성 - LangChain 체인 기능 활용."""
    try:
        # 1. Retriever 생성 (현재 Embedding 모델 사용)
        current_vector_store = PGVector(
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
        )
        retriever = current_vector_store.as_retriever(search_kwargs={"k": 3})

        # 2. 대화 기록을 고려한 검색 쿼리 생성 프롬프트
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

        # 3. 대화 기록을 고려한 Retriever 생성
        history_aware_retriever = create_history_aware_retriever(
            llm_model, retriever, contextualize_q_prompt
        )

        # 4. 질문 답변 프롬프트
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

        # 5. 문서 결합 체인 생성
        question_answer_chain = create_stuff_documents_chain(llm_model, qa_prompt)

        # 6. 최종 RAG 체인 생성
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return rag_chain
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] RAG 체인 생성 실패: {error_msg[:200]}...")
        raise


def initialize_rag_chain():
    """RAG 체인 초기화 - OpenAI와 로컬 모델용 체인 생성."""
    global openai_rag_chain, local_rag_chain

    # OpenAI용 RAG 체인 생성
    if openai_llm and openai_embeddings:
        try:
            openai_rag_chain = create_rag_chain(openai_llm, openai_embeddings)
            print("[OK] OpenAI RAG 체인 초기화 완료")
        except Exception as e:
            print(f"[WARNING] OpenAI RAG 체인 초기화 실패: {str(e)[:100]}...")
            openai_rag_chain = None

    # 로컬 모델용 RAG 체인 생성
    if local_llm and local_embeddings:
        try:
            local_rag_chain = create_rag_chain(local_llm, local_embeddings)
            print("[OK] 로컬 RAG 체인 초기화 완료")
        except Exception as e:
            print(f"[WARNING] 로컬 RAG 체인 초기화 실패: {str(e)[:100]}...")
            local_rag_chain = None

    if not openai_rag_chain and not local_rag_chain:
        error_msg = "OpenAI와 로컬 RAG 체인 모두 초기화에 실패했습니다.\n"
        if not openai_llm:
            error_msg += "- OpenAI LLM이 초기화되지 않았습니다.\n"
        if not openai_embeddings:
            error_msg += "- OpenAI Embeddings가 초기화되지 않았습니다.\n"
        if not local_llm:
            error_msg += "- 로컬 LLM이 초기화되지 않았습니다.\n"
        if not local_embeddings:
            error_msg += "- 로컬 Embeddings가 초기화되지 않았습니다.\n"
        raise RuntimeError(error_msg)


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화."""
    print("=" * 50)
    print("LangChain FastAPI 서버 시작 중...")
    print("=" * 50)

    # 환경 변수 확인
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    local_model_dir = os.getenv("LOCAL_MODEL_DIR", "기본값 사용")
    print(f"\n[INFO] LLM_PROVIDER: {llm_provider}")
    print(f"[INFO] LOCAL_MODEL_DIR: {local_model_dir}")

    # Neon PostgreSQL 연결 대기
    print("\n1. Neon PostgreSQL 연결 확인 중...")
    wait_for_postgres()

    # Embedding 모델 초기화
    print("\n2. Embedding 모델 초기화 중...")
    initialize_embeddings()

    # LLM 모델 초기화
    print("\n3. LLM 모델 초기화 중...")
    initialize_llm()

    # PGVector 스토어 초기화
    print("\n4. PGVector 스토어 초기화 중...")
    initialize_vector_store()

    # RAG 체인 초기화
    print("\n5. RAG 체인 초기화 중...")
    initialize_rag_chain()

    print("\n" + "=" * 50)
    print("[OK] 서버 초기화 완료!")
    print("=" * 50)


# 요청/응답 모델
class ChatRequest(BaseModel):
    """챗봇 요청 모델."""

    message: str
    history: Optional[List[dict]] = []
    model_type: Optional[str] = "openai"  # "openai" 또는 "local"


class ChatResponse(BaseModel):
    """챗봇 응답 모델."""

    response: str


@app.get("/")
async def root():
    """루트 엔드포인트."""
    return {
        "message": "LangChain Chatbot API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트."""
    global openai_quota_exceeded
    return {
        "status": "healthy",
        "vector_store": "initialized" if vector_store else "not initialized",
        "openai_embeddings": "initialized" if openai_embeddings else "not initialized",
        "local_embeddings": "initialized" if local_embeddings else "not initialized",
        "openai_llm": "initialized" if openai_llm else "not initialized",
        "local_llm": "initialized" if local_llm else "not initialized",
        "openai_rag_chain": "initialized" if openai_rag_chain else "not initialized",
        "local_rag_chain": "initialized" if local_rag_chain else "not initialized",
        "openai_quota_exceeded": openai_quota_exceeded,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """챗봇 API 엔드포인트 - LangChain RAG 체인 사용."""
    # 모델 타입에 따라 적절한 RAG 체인 선택
    # 프론트엔드에서 전달된 model_type이 없으면 .env의 LLM_PROVIDER 사용
    model_type = request.model_type or os.getenv("LLM_PROVIDER", "openai")
    if model_type:
        model_type = model_type.lower()

    # 디버깅: 받은 model_type 로그 출력
    print(
        f"[DEBUG] 받은 model_type: {request.model_type}, 처리된 model_type: {model_type}"
    )

    # "midm"도 "local"로 처리
    if model_type == "midm":
        model_type = "local"

    if model_type == "openai":
        if not openai_rag_chain:
            # 할당량 초과 여부 확인
            global openai_quota_exceeded

            if openai_quota_exceeded:
                # 할당량 초과인 경우 명확한 메시지
                error_msg = (
                    "⚠️ OpenAI API 할당량이 초과되었습니다.\n\n"
                    "서버 시작 시 '[WARNING] OpenAI API 할당량 초과' 메시지가 확인되었습니다.\n\n"
                    "해결 방법:\n"
                    "1. OpenAI 계정의 사용량 및 할당량을 확인하세요\n"
                    "2. OpenAI 계정에 결제 정보를 추가하거나 할당량을 늘리세요\n"
                    "3. 또는 '🖥️ 로컬 모델' 버튼을 선택하여 로컬 Midm 모델을 사용하세요"
                )
            elif not openai_llm and not openai_embeddings:
                # 둘 다 초기화 실패 (할당량 초과가 아닌 경우)
                error_msg = (
                    "OpenAI 모델이 초기화되지 않았습니다.\n\n"
                    "가능한 원인:\n"
                    "1. OpenAI API 키가 설정되지 않았거나 잘못되었습니다\n"
                    "2. 네트워크 연결 문제\n\n"
                    "해결 방법:\n"
                    "- .env 파일에 올바른 OPENAI_API_KEY를 설정하세요\n"
                    "- 또는 '로컬 모델' 버튼을 선택하여 로컬 모델을 사용하세요"
                )
            else:
                # 일부만 실패
                error_details = []
                if not openai_llm:
                    error_details.append("OpenAI LLM이 초기화되지 않았습니다")
                if not openai_embeddings:
                    error_details.append("OpenAI Embeddings가 초기화되지 않았습니다")
                error_msg = f"OpenAI RAG 체인 생성 실패: {', '.join(error_details)}"

            print(f"[ERROR] OpenAI 모델 사용 시도 실패: {error_msg}")
            raise HTTPException(
                status_code=503,
                detail=error_msg,
            )
        current_rag_chain = openai_rag_chain
    elif model_type == "local" or model_type == "midm":
        if not local_rag_chain:
            raise HTTPException(
                status_code=503,
                detail="로컬 모델이 초기화되지 않았습니다. Midm 모델과 sentence-transformers를 확인해주세요.",
            )
        print(f"[DEBUG] 로컬 RAG 체인 사용 (model_type: {model_type})")
        current_rag_chain = local_rag_chain
    else:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 모델 타입입니다: {model_type}. 'openai' 또는 'local'을 사용해주세요.",
        )

    try:
        # 대화 기록을 LangChain 메시지 형식으로 변환
        chat_history = []
        if request.history:
            for msg in request.history:
                if msg.get("role") == "user":
                    chat_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    chat_history.append(AIMessage(content=msg.get("content", "")))

        # RAG 체인 실행
        result = current_rag_chain.invoke(
            {
                "input": request.message,
                "chat_history": chat_history,
            }
        )

        # 체인 결과에서 답변 추출
        response_text = result.get("answer", "답변을 생성할 수 없습니다.")

        # response_text가 None이거나 문자열이 아닌 경우 처리
        if response_text is None:
            response_text = "답변을 생성할 수 없습니다."
        else:
            response_text = str(response_text)

        # 응답에서 이전 대화 내용 제거 (중복 방지)
        # Midm 모델에서 이미 정리했으므로 간단한 체크만 수행
        if response_text and (
            "Human:" in response_text or "Assistant:" in response_text
        ):
            import re

            # 빠른 정규식으로 마지막 Assistant: 이후만 추출
            assistant_match = re.search(
                r"Assistant:\s*(.+?)(?:\nHuman:|$)", response_text, re.DOTALL
            )
            if assistant_match:
                response_text = assistant_match.group(1).strip()

        # 빈 응답 방지
        if not response_text or not response_text.strip():
            response_text = "답변을 생성할 수 없습니다."

        return ChatResponse(response=response_text)

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] 챗봇 응답 생성 실패: {error_msg}")

        # OpenAI API 호출량 초과 에러 확인
        if (
            "quota" in error_msg.lower()
            or "429" in error_msg
            or "insufficient_quota" in error_msg
            or "exceeded" in error_msg.lower()
        ):
            error_detail = "OpenAI API 호출량이 초과되었습니다. 할당량을 확인하고 다시 시도해주세요."
            raise HTTPException(
                status_code=429,
                detail=error_detail,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"응답 생성 중 오류가 발생했습니다: {error_msg[:200]}",
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
