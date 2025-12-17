"""Neon DB 초기 문서 저장 확인 스크립트."""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
from dotenv import load_dotenv

load_dotenv()

import psycopg2

from api.app.api_server import COLLECTION_NAME, CONNECTION_STRING

print("=" * 70)
print("Neon DB 초기 문서 저장 확인")
print("=" * 70)

try:
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    # 1. 모든 컬렉션 확인
    print("\n[1] 모든 컬렉션 목록")
    print("-" * 70)
    cur.execute("SELECT uuid, name FROM langchain_pg_collection")
    all_collections = cur.fetchall()
    if all_collections:
        print(f"총 {len(all_collections)}개 컬렉션 발견:")
        for col_uuid, col_name in all_collections:
            print(f"  - {col_name} (UUID: {col_uuid})")

            # 각 컬렉션의 벡터 개수
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM langchain_pg_embedding
                WHERE collection_id = '{col_uuid}'
            """
            )
            count = cur.fetchone()[0]
            print(f"    벡터 개수: {count}")
    else:
        print("컬렉션이 없습니다.")

    # 2. 특정 컬렉션 확인
    print(f"\n[2] '{COLLECTION_NAME}' 컬렉션 상세 확인")
    print("-" * 70)
    cur.execute(
        f"""
        SELECT uuid, name, cmetadata
        FROM langchain_pg_collection
        WHERE name = '{COLLECTION_NAME}'
    """
    )
    collection = cur.fetchone()
    if collection:
        collection_uuid = collection[0]
        print(f"컬렉션 UUID: {collection_uuid}")
        print(f"컬렉션 이름: {collection[1]}")

        # 벡터 개수
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM langchain_pg_embedding
            WHERE collection_id = '{collection_uuid}'
        """
        )
        vector_count = cur.fetchone()[0]
        print(f"벡터 개수: {vector_count}")

        # 저장된 문서 내용 확인
        if vector_count > 0:
            print("\n[3] 저장된 문서 내용")
            print("-" * 70)
            cur.execute(
                f"""
                SELECT document, cmetadata
                FROM langchain_pg_embedding
                WHERE collection_id = '{collection_uuid}'
                ORDER BY uuid
            """
            )
            documents = cur.fetchall()
            for i, (doc, metadata) in enumerate(documents, 1):
                print(f"\n[{i}] {doc}")
                if metadata:
                    print(f"    메타데이터: {metadata}")
        else:
            print("\n[정보] 벡터 데이터가 없습니다.")
            print("서버가 아직 초기화되지 않았거나, 초기 문서가 저장되지 않았습니다.")

            # 초기 문서가 저장되어야 하는 내용
            print("\n[예상 초기 문서]")
            print("-" * 70)
            expected_docs = [
                "LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.",
                "pgvector는 PostgreSQL에서 벡터 검색을 가능하게 하는 확장입니다.",
                "Hello World는 프로그래밍의 첫 번째 예제입니다.",
            ]
            for i, doc in enumerate(expected_docs, 1):
                print(f"{i}. {doc}")
    else:
        print(f"[경고] '{COLLECTION_NAME}' 컬렉션이 존재하지 않습니다.")
        print("서버를 시작하면 자동으로 생성됩니다.")

    # 4. 테이블 구조 확인
    print("\n[4] 테이블 구조 확인")
    print("-" * 70)
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name LIKE 'langchain%'
        ORDER BY table_name
    """)
    tables = cur.fetchall()
    print("LangChain 관련 테이블:")
    for table in tables:
        print(f"  - {table[0]}")

    print("\n" + "=" * 70)
    print("[결론]")
    print("=" * 70)
    if collection and vector_count == 0:
        print("[WARNING] 컬렉션은 존재하지만 벡터 데이터가 없습니다.")
        print("서버를 재시작하면 초기 문서 3개가 자동으로 저장됩니다.")
    elif not collection:
        print("[WARNING] 컬렉션이 아직 생성되지 않았습니다.")
        print("서버를 시작하면 자동으로 생성됩니다.")
    else:
        print(
            f"[OK] 컬렉션과 벡터 데이터가 정상적으로 존재합니다. (벡터: {vector_count}개)"
        )
    print("=" * 70)

    conn.close()

except psycopg2.OperationalError as e:
    print(f"\n[오류] PostgreSQL 연결 실패: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[오류] 오류 발생: {e}")
    import traceback

    print(traceback.format_exc())
    sys.exit(1)
