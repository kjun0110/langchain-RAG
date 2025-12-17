"""
메인 진입점 - 라우터 역할.

환경 변수 MODE에 따라 worker 또는 api 서버를 실행합니다.
- MODE=worker: LangChain과 pgvector 연결 워커 실행
- MODE=api 또는 미설정: FastAPI 서버 실행
"""

import logging
import os
import sys
from pathlib import Path

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

# 현재 파일의 디렉토리를 Python 경로에 추가
# 이렇게 하면 같은 디렉토리의 모듈을 import할 수 있습니다
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Logging 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
    ],
)

# 실행 모드 확인
MODE = os.getenv("MODE", "api").lower()

if MODE == "worker":
    # Worker 모드: LangChain과 pgvector 연결 워커 실행
    logging.info("Starting LangChain pgvector worker...")
    # app.app 모듈에서 main 함수 실행 (PYTHONPATH=/app이므로 app.app이 /app/app/app.py를 가리킴)
    try:
        from app import app as worker_app
    except ImportError:
        # 상대 import 시도 (같은 패키지 내)
        from . import app as worker_app

    worker_app.main()
elif MODE == "api":
    # API 모드: FastAPI 서버 실행
    logging.info("Starting FastAPI server...")
    try:
        from app import api_server
    except ImportError:
        # 상대 import 시도 (같은 패키지 내)
        from . import api_server
    import uvicorn

    # 로컬 실행 시 host를 127.0.0.1로 변경 (0.0.0.0은 모든 인터페이스)
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(api_server.app, host=host, port=port)
else:
    logging.error(f"Unknown MODE: {MODE}. Use 'worker' or 'api'")
    sys.exit(1)
