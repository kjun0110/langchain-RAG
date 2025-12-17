"""
LangChain과 pgvector 연결 워커.

이 스크립트는 LangChain과 PostgreSQL pgvector 확장을 연결하여
벡터 스토어를 초기화하고 모니터링하는 역할을 합니다.

역할:
- PostgreSQL pgvector와의 연결 관리
- 벡터 스토어 초기화 및 데이터 관리
- 벡터 검색 기능 테스트 및 모니터링
"""

import os
import time
import warnings
from pathlib import Path
from typing import List

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
    # langchain_core가 없는 경우 일반 DeprecationWarning 무시
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="langchain_community.vectorstores.pgvector",
    )

# 일반 DeprecationWarning도 무시
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="langchain_community.vectorstores.pgvector",
)

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings

# .env 파일 로드
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv가 없으면 환경 변수만 사용

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


def wait_for_postgres(max_retries: int = 30, delay: int = 2) -> None:
    """Neon PostgreSQL이 준비될 때까지 대기."""
    import psycopg2

    for i in range(max_retries):
        try:
            conn = psycopg2.connect(CONNECTION_STRING)
            conn.close()
            print("✓ Neon PostgreSQL 연결 성공!")
            return
        except Exception as e:
            if i < max_retries - 1:
                print(f"Neon PostgreSQL 대기 중... ({i + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise ConnectionError(f"Neon PostgreSQL 연결 실패: {e}")


def main() -> None:
    """LangChain Hello World 메인 함수."""
    print("=" * 50)
    print("LangChain Hello World with pgvector")
    print("=" * 50)

    # Neon PostgreSQL 연결 대기
    print("\n1. Neon PostgreSQL 연결 확인 중...")
    wait_for_postgres()

    # Embedding 모델 초기화 (OpenAI API 키가 있으면 사용, 없으면 FakeEmbeddings 사용)
    print("\n2. Embedding 모델 초기화 중...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = None

    if openai_api_key and openai_api_key != "your-api-key-here":
        try:
            # OpenAIEmbeddings 초기화 및 간단한 테스트
            test_embeddings = OpenAIEmbeddings()
            # 실제 API 호출 테스트 (할당량 확인)
            test_embeddings.embed_query("test")
            embeddings = test_embeddings
            print("✓ OpenAI Embedding 모델 초기화 완료")
        except Exception as e:
            error_msg = str(e)
            if (
                "quota" in error_msg.lower()
                or "429" in error_msg
                or "insufficient_quota" in error_msg
            ):
                print(f"⚠ OpenAI API 할당량 초과: {error_msg[:100]}...")
            else:
                print(f"⚠ OpenAI Embedding 모델 초기화 실패: {error_msg[:100]}...")
            print("   FakeEmbeddings로 대체합니다...")
            embeddings = FakeEmbeddings(size=1536)
            print("✓ FakeEmbeddings 초기화 완료")
    else:
        print("   OpenAI API 키가 없습니다. FakeEmbeddings를 사용합니다.")
        embeddings = FakeEmbeddings(size=1536)
        print("✓ FakeEmbeddings 초기화 완료")

    # PGVector 스토어 생성
    print("\n3. PGVector 스토어 생성 중...")
    try:
        vector_store = PGVector.from_documents(
            embedding=embeddings,
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
        print("✓ PGVector 스토어 생성 완료")
    except Exception as e:
        error_msg = str(e)
        # OpenAI API 오류인 경우 FakeEmbeddings로 재시도
        if (
            "quota" in error_msg.lower()
            or "429" in error_msg
            or "insufficient_quota" in error_msg
        ):
            print("⚠ OpenAI API 할당량 초과로 인한 오류 발생")
            print("   FakeEmbeddings로 재시도합니다...")
            try:
                embeddings = FakeEmbeddings(size=1536)
                vector_store = PGVector.from_documents(
                    embedding=embeddings,
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
                print("✓ FakeEmbeddings로 PGVector 스토어 생성 완료")
            except Exception as retry_error:
                print(f"✗ FakeEmbeddings로도 PGVector 스토어 생성 실패: {retry_error}")
                return
        else:
            print(f"✗ PGVector 스토어 생성 실패: {error_msg[:200]}...")
            return

    # 벡터 검색 테스트
    print("\n4. 벡터 검색 테스트 중...")
    try:
        query = "프레임워크"
        results: List[Document] = vector_store.similarity_search(query, k=2)

        print(f"\n검색 쿼리: '{query}'")
        print(f"검색 결과 ({len(results)}개):")
        for i, doc in enumerate(results, 1):
            print(f"\n  [{i}] {doc.page_content}")
            print(f"      메타데이터: {doc.metadata}")
    except Exception as e:
        print(f"✗ 벡터 검색 실패: {e}")
        return

    print("\n" + "=" * 50)
    print("✓ LangChain Hello World 완료!")
    print("=" * 50)

    # 컨테이너가 계속 실행되도록 대기 및 PGVector 데이터 주기적 조회
    print("\n" + "=" * 50)
    print("PGVector 연결 모니터링 시작... (종료하려면 Ctrl+C)")
    print("=" * 50)

    check_count = 0
    try:
        while True:
            time.sleep(30)  # 30초마다 데이터 조회
            check_count += 1

            print(f"\n[{check_count}] PGVector 데이터 조회 중...")
            print("-" * 50)

            try:
                # PGVector에서 모든 문서 조회 (더미 쿼리로 전체 데이터 가져오기)
                # similarity_search를 사용하여 다양한 쿼리로 데이터 조회
                test_queries = ["LangChain", "pgvector", "Hello", "프레임워크"]

                for query in test_queries:
                    try:
                        query_results: List[Document] = vector_store.similarity_search(
                            query, k=3
                        )
                        print(f"\n  쿼리: '{query}' → {len(query_results)}개 결과")
                        for i, doc in enumerate(
                            query_results[:2], 1
                        ):  # 최대 2개만 출력
                            print(f"    [{i}] {doc.page_content[:60]}...")
                            print(f"        메타데이터: {doc.metadata}")
                    except Exception as e:
                        print(f"    ⚠ 쿼리 '{query}' 실패: {e}")

                # Neon PostgreSQL에서 직접 데이터 개수 확인
                try:
                    import psycopg2

                    conn = psycopg2.connect(CONNECTION_STRING)
                    cursor = conn.cursor()

                    # PGVector 테이블 구조 확인 및 데이터 개수 조회
                    # langchain_pg_embedding과 langchain_pg_collection 테이블 사용
                    try:
                        cursor.execute(
                            """
                            SELECT COUNT(*)
                            FROM langchain_pg_embedding
                            WHERE collection_id = (
                                SELECT uuid FROM langchain_pg_collection WHERE name = %s
                            )
                            """,
                            (COLLECTION_NAME,),
                        )
                        result = cursor.fetchone()
                        count = result[0] if result else 0
                    except Exception:
                        # 테이블 이름이 다를 수 있으므로 다른 방법 시도
                        cursor.execute(
                            "SELECT COUNT(*) FROM information_schema.tables "
                            "WHERE table_schema = 'public' AND table_name LIKE '%embedding%'"
                        )
                        result = cursor.fetchone()
                        count = result[0] if result else "확인 불가"

                    cursor.close()
                    conn.close()

                    print("\n  📊 PGVector 저장소 통계:")
                    print(f"     - 컬렉션: {COLLECTION_NAME}")
                    print(f"     - 저장된 문서 수: {count}개")
                    print("     - PostgreSQL 연결: ✓ 정상")
                except Exception as db_error:
                    print(f"\n  📊 Neon PostgreSQL 직접 조회 실패: {db_error}")
                    print("     - PGVector 검색은 정상 작동 중")

                print("-" * 50)

            except Exception as e:
                print(f"  ✗ PGVector 데이터 조회 실패: {e}")
                print(f"     오류 타입: {type(e).__name__}")

    except KeyboardInterrupt:
        print("\n\n컨테이너 종료 중...")


if __name__ == "__main__":
    main()
