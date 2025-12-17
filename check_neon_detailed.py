"""Neon PGVector 상세 연결 확인 스크립트."""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
from dotenv import load_dotenv

load_dotenv()

# api_server에서 연결 정보 가져오기
import psycopg2

from api.app.api_server import COLLECTION_NAME, CONNECTION_STRING

print("=" * 60)
print("Neon PGVector 상세 연결 확인")
print("=" * 60)

try:
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    # 1. 데이터베이스 정보
    print("\n[1] 데이터베이스 정보")
    print("-" * 60)
    cur.execute("SELECT current_database(), version()")
    db_info = cur.fetchone()
    print(f"데이터베이스: {db_info[0]}")
    print(f"PostgreSQL 버전: {db_info[1].split(',')[0]}")

    # 2. PGVector 확장 상세 정보
    print("\n[2] PGVector 확장 정보")
    print("-" * 60)
    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
    result = cur.fetchone()
    if result:
        print(f"확장 이름: {result[0]}")
        print(f"버전: {result[1]}")

        # 벡터 함수 확인
        cur.execute("""
            SELECT proname
            FROM pg_proc
            WHERE proname LIKE '%vector%'
            LIMIT 5
        """)
        vector_funcs = cur.fetchall()
        print(f"벡터 관련 함수 예시: {[f[0] for f in vector_funcs]}")
    else:
        print("[경고] PGVector 확장이 설치되지 않았습니다!")

    # 3. LangChain 테이블 구조 확인
    print("\n[3] LangChain 테이블 구조")
    print("-" * 60)

    # langchain_pg_collection 테이블
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'langchain_pg_collection'
        ORDER BY ordinal_position
    """)
    collection_columns = cur.fetchall()
    print("langchain_pg_collection 컬럼:")
    for col in collection_columns:
        print(f"  - {col[0]}: {col[1]}")

    # langchain_pg_embedding 테이블
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'langchain_pg_embedding'
        ORDER BY ordinal_position
    """)
    embedding_columns = cur.fetchall()
    print("\nlangchain_pg_embedding 컬럼:")
    for col in embedding_columns:
        print(f"  - {col[0]}: {col[1]}")

    # 4. 컬렉션 정보
    print("\n[4] 컬렉션 정보")
    print("-" * 60)
    cur.execute(f"""
        SELECT uuid, name, cmetadata
        FROM langchain_pg_collection
        WHERE name = '{COLLECTION_NAME}'
    """)
    collection = cur.fetchone()
    if collection:
        print(f"컬렉션 UUID: {collection[0]}")
        print(f"컬렉션 이름: {collection[1]}")
        print(f"메타데이터: {collection[2]}")

        # 해당 컬렉션의 벡터 개수
        cur.execute(f"""
            SELECT COUNT(*)
            FROM langchain_pg_embedding
            WHERE collection_id = '{collection[0]}'
        """)
        embedding_count = cur.fetchone()[0]
        print(f"벡터 임베딩 개수: {embedding_count}")

        # 샘플 벡터 확인
        if embedding_count > 0:
            cur.execute(f"""
                SELECT id, document, embedding
                FROM langchain_pg_embedding
                WHERE collection_id = '{collection[0]}'
                LIMIT 1
            """)
            sample = cur.fetchone()
            if sample:
                print("\n샘플 데이터:")
                print(f"  ID: {sample[0]}")
                print(f"  문서: {sample[1][:100]}...")
                print(f"  벡터 차원: {len(sample[2]) if sample[2] else 0}")
    else:
        print(f"[정보] 컬렉션 '{COLLECTION_NAME}'가 아직 생성되지 않았습니다.")

    # 5. 모든 컬렉션 목록
    print("\n[5] 모든 컬렉션 목록")
    print("-" * 60)
    cur.execute("SELECT name FROM langchain_pg_collection")
    all_collections = cur.fetchall()
    if all_collections:
        print("컬렉션 목록:")
        for col in all_collections:
            print(f"  - {col[0]}")
    else:
        print("컬렉션이 없습니다.")

    # 6. 연결 문자열 정보 (보안을 위해 일부만 표시)
    print("\n[6] 연결 정보")
    print("-" * 60)
    # 연결 문자열에서 호스트만 추출
    if "@" in CONNECTION_STRING and "/" in CONNECTION_STRING:
        host_part = CONNECTION_STRING.split("@")[1].split("/")[0]
        print(f"호스트: {host_part}")
    print(f"컬렉션 이름: {COLLECTION_NAME}")

    print("\n" + "=" * 60)
    print("[결론] Neon PGVector 연결 상태: 정상")
    print("=" * 60)

    conn.close()

except psycopg2.OperationalError as e:
    print(f"\n[오류] PostgreSQL 연결 실패: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[오류] 오류 발생: {e}")
    import traceback

    print(traceback.format_exc())
    sys.exit(1)
