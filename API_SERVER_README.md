# FastAPI 백엔드 서버 - LangChain 연동 가이드

## 개요

`api_server.py`는 Frontend와 LangChain을 연동하는 FastAPI 백엔드 서버입니다.

## 기능

- ✅ PGVector 벡터 검색 연동
- ✅ LangChain RAG (Retrieval-Augmented Generation)
- ✅ 대화 기록 지원
- ✅ OpenAI 또는 FakeEmbeddings/FakeChatModel 지원
- ✅ CORS 설정 (Frontend 연동)
- ✅ 자동 API 문서화 (Swagger UI)

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성 (선택사항):

```env
OPENAI_API_KEY=your-api-key-here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=langchain
POSTGRES_PASSWORD=langchain123
POSTGRES_DB=langchain_db
```

## 실행 방법

### 방법 1: 직접 실행

```bash
python api_server.py
```

### 방법 2: uvicorn으로 실행

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

서버가 시작되면:
- API 서버: http://localhost:8000
- API 문서: http://localhost:8000/docs
- 헬스 체크: http://localhost:8000/health

## API 엔드포인트

### POST /api/chat

챗봇과 대화하는 엔드포인트입니다.

**요청:**
```json
{
  "message": "LangChain이 뭐야?",
  "history": [
    {"role": "user", "content": "안녕하세요"},
    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
  ]
}
```

**응답:**
```json
{
  "response": "LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다..."
}
```

### GET /health

서버 상태 확인 엔드포인트입니다.

**응답:**
```json
{
  "status": "healthy",
  "vector_store": "initialized",
  "embeddings": "initialized",
  "llm": "initialized"
}
```

## Frontend 연동

Frontend는 이미 `http://localhost:8000/api/chat`를 호출하도록 설정되어 있습니다.

1. FastAPI 서버 실행 (`python api_server.py`)
2. Frontend 실행 (`cd frontend && pnpm dev`)
3. 브라우저에서 http://localhost:3000 접속

## Docker Compose 사용

`docker-compose.yaml`을 수정하여 `api_server.py`를 실행하도록 변경할 수 있습니다:

```yaml
langchain-app:
  # ... 기존 설정 ...
  command: python api_server.py  # app.py 대신 api_server.py
```

또는 uvicorn 사용:

```yaml
command: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 동작 원리

1. **사용자 메시지 수신**: Frontend에서 `/api/chat`로 요청
2. **벡터 검색**: PGVector에서 관련 문서 검색 (similarity_search)
3. **컨텍스트 구성**: 검색된 문서를 컨텍스트로 사용
4. **LLM 응답 생성**: LangChain LLM이 컨텍스트와 대화 기록을 바탕으로 응답 생성
5. **응답 반환**: 생성된 응답을 Frontend로 반환

## 문제 해결

### PostgreSQL 연결 실패
- PostgreSQL이 실행 중인지 확인
- 환경 변수 설정 확인
- `docker-compose up postgres`로 PostgreSQL 시작

### OpenAI API 오류
- API 키가 올바른지 확인
- 할당량 확인
- FakeEmbeddings/FakeChatModel로 자동 대체됨

### CORS 오류
- `api_server.py`의 CORS 설정 확인
- Frontend URL을 `allow_origins`에 추가

## 참고

- API 문서는 http://localhost:8000/docs 에서 확인 가능
- Swagger UI에서 직접 API 테스트 가능

