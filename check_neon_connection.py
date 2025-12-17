"""Neon PGVector 연결 확인 스크립트."""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
from dotenv import load_dotenv

load_dotenv()

# api_server에서 연결 정보 가져오기
from api.app.api_server import CONNECTION_STRING, COLLECTION_NAME

import psycopg2

print("=" * 50)
print("Neon PGVector 연결 확인")
print("=" * 50)

try:
    # 1. PostgreSQL 연결 테스트
    print("\n1. PostgreSQL 연결 테스트 중...")
    conn = psycopg2.connect(CONNECTION_STRING)
    print("[OK] PostgreSQL 연결 성공!")

    cur = conn.cursor()

    # 2. PGVector 확장 확인
    print("\n2. PGVector 확장 확인 중...")
    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
    result = cur.fetchone()
    if result:
        print(f"[OK] PGVector 확장 설치됨 (버전: {result[1]})")
    else:
        print("[WARNING] PGVector 확장이 설치되지 않았습니다.")

    # 3. LangChain 컬렉션 테이블 확인
    print("\n3. LangChain 컬렉션 테이블 확인 중...")
    cur.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE '%langchain%'"
    )
    tables = cur.fetchall()
    if tables:
        print(f"[OK] LangChain 관련 테이블 발견: {[t[0] for t in tables]}")
    else:
        print("[INFO] LangChain 관련 테이블이 없습니다 (초기 문서 생성 시 자동 생성됨)")

    # 4. 컬렉션 데이터 확인
    print(f"\n4. 컬렉션 '{COLLECTION_NAME}' 데이터 확인 중...")
    cur.execute(
        f"SELECT COUNT(*) FROM langchain_pg_collection WHERE name = '{COLLECTION_NAME}'"
    )
    collection_count = cur.fetchone()[0]
    if collection_count > 0:
        print(f"[OK] 컬렉션 '{COLLECTION_NAME}' 존재함")

        # 벡터 데이터 개수 확인
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection WHERE name = '{COLLECTION_NAME}'
            )
            """
        )
        embedding_count = cur.fetchone()[0]
        print(f"[INFO] 벡터 임베딩 개수: {embedding_count}")
    else:
        print(f"[INFO] 컬렉션 '{COLLECTION_NAME}'가 아직 생성되지 않았습니다.")

    # 5. 연결 정보 요약
    print("\n" + "=" * 50)
    print("연결 정보 요약")
    print("=" * 50)
    print(f"연결 문자열: {CONNECTION_STRING[:50]}...")
    print(f"컬렉션 이름: {COLLECTION_NAME}")
    print(f"PGVector 확장: {'설치됨' if result else '설치 안됨'}")
    print(f"컬렉션 존재: {'예' if collection_count > 0 else '아니오'}")
    print("=" * 50)

    conn.close()
    print("\n[OK] 모든 확인 완료!")

except psycopg2.OperationalError as e:
    print(f"\n[ERROR] PostgreSQL 연결 실패: {e}")
    print("\n가능한 원인:")
    print("1. .env 파일에 DATABASE_URL이 올바르게 설정되지 않았습니다")
    print("2. 네트워크 연결 문제")
    print("3. Neon 데이터베이스가 일시 중지되었습니다")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] 오류 발생: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

