"""Neon DB의 벡터 데이터 확인 스크립트."""

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
print("Neon DB 벡터 데이터 확인")
print("=" * 70)

try:
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    # 1. 컬렉션 정보
    print(f"\n[1] 컬렉션 '{COLLECTION_NAME}' 정보")
    print("-" * 70)
    cur.execute(
        f"""
        SELECT uuid, name, cmetadata
        FROM langchain_pg_collection
        WHERE name = '{COLLECTION_NAME}'
    """
    )
    collection = cur.fetchone()
    if not collection:
        print(f"[경고] 컬렉션 '{COLLECTION_NAME}'가 존재하지 않습니다.")
        conn.close()
        sys.exit(0)

    collection_uuid = collection[0]
    print(f"컬렉션 UUID: {collection_uuid}")
    print(f"컬렉션 이름: {collection[1]}")
    print(f"메타데이터: {collection[2]}")

    # 2. 벡터 데이터 통계
    print("\n[2] 벡터 데이터 통계")
    print("-" * 70)
    cur.execute(
        f"""
        SELECT COUNT(*) as total,
               COUNT(DISTINCT document) as unique_documents
        FROM langchain_pg_embedding
        WHERE collection_id = '{collection_uuid}'
    """
    )
    stats = cur.fetchone()
    total_count = stats[0]
    unique_docs = stats[1]

    print(f"총 벡터 개수: {total_count}")
    print(f"고유 문서 개수: {unique_docs}")

    if total_count == 0:
        print("\n[정보] 벡터 데이터가 없습니다.")
        print("서버 초기화 시 기본 문서 3개가 자동으로 저장됩니다.")
        print(
            "챗봇을 사용하면 관련 문서가 검색되지만, 대화 내용은 자동으로 저장되지 않습니다."
        )
        conn.close()
        sys.exit(0)

    # 3. 벡터 데이터 상세 정보
    print("\n[3] 저장된 벡터 데이터 목록")
    print("-" * 70)
    cur.execute(
        f"""
        SELECT uuid, document,
               LENGTH(embedding::text) as embedding_size,
               cmetadata
        FROM langchain_pg_embedding
        WHERE collection_id = '{collection_uuid}'
        ORDER BY uuid
        LIMIT 20
    """
    )
    vectors = cur.fetchall()

    for i, (vec_uuid, document, emb_size, metadata) in enumerate(vectors, 1):
        print(f"\n[{i}] 벡터 UUID: {vec_uuid}")
        doc_preview = document[:100] + "..." if len(document) > 100 else document
        print(f"    문서: {doc_preview}")
        print(f"    임베딩 크기: {emb_size} bytes")
        if metadata:
            print(f"    메타데이터: {metadata}")

    if total_count > 20:
        print(f"\n... 외 {total_count - 20}개 더 있음")

    # 4. 문서별 통계
    print("\n[4] 문서별 통계")
    print("-" * 70)
    cur.execute(
        f"""
        SELECT
            LEFT(document, 50) as doc_preview,
            COUNT(*) as chunk_count
        FROM langchain_pg_embedding
        WHERE collection_id = '{collection_uuid}'
        GROUP BY document
        ORDER BY chunk_count DESC
        LIMIT 10
    """
    )
    doc_stats = cur.fetchall()
    if doc_stats:
        print("문서별 청크 개수:")
        for doc_preview, chunk_count in doc_stats:
            print(f"  - {doc_preview}... : {chunk_count}개 청크")

    # 5. 벡터 차원 확인
    print("\n[5] 벡터 차원 정보")
    print("-" * 70)
    cur.execute(
        f"""
        SELECT
            array_length(embedding::vector, 1) as vector_dimension
        FROM langchain_pg_embedding
        WHERE collection_id = '{collection_uuid}'
        LIMIT 1
    """
    )
    dim_result = cur.fetchone()
    if dim_result and dim_result[0]:
        print(f"벡터 차원: {dim_result[0]}차원")
    else:
        print("벡터 차원을 확인할 수 없습니다.")

    # 6. 최근 추가된 벡터 (시간 정보가 있다면)
    print("\n[6] 메타데이터 분석")
    print("-" * 70)
    cur.execute(
        f"""
        SELECT cmetadata
        FROM langchain_pg_embedding
        WHERE collection_id = '{collection_uuid}'
          AND cmetadata IS NOT NULL
        LIMIT 5
    """
    )
    metadata_samples = cur.fetchall()
    if metadata_samples:
        print("메타데이터 샘플:")
        for meta in metadata_samples:
            print(f"  - {meta[0]}")
    else:
        print("메타데이터가 없습니다.")

    # 7. 요약
    print("\n" + "=" * 70)
    print("[요약]")
    print("=" * 70)
    print(f"컬렉션: {COLLECTION_NAME}")
    print(f"총 벡터 개수: {total_count}")
    print(f"고유 문서 개수: {unique_docs}")
    if dim_result and dim_result[0]:
        print(f"벡터 차원: {dim_result[0]}차원")
    print("\n[참고]")
    print("- 현재는 서버 초기화 시 기본 문서만 저장됩니다.")
    print("- 챗봇 대화 내용은 자동으로 벡터 스토어에 저장되지 않습니다.")
    print("- 대화 내용을 저장하려면 추가 기능 구현이 필요합니다.")
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
