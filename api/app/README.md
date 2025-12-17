# LLM 모델 주입 아키텍처

## 폴더 구조

```
api/app/
├── model/                    # LLM 모델 인터페이스 및 추상화
│   ├── base.py              # BaseLLM, BaseEmbedding 인터페이스
│   ├── types.py             # 타입 정의
│   ├── factory.py           # 모델 팩토리 (등록/조회)
│   └── implementations/     # 모델 구현체 (사용자가 추가)
│       └── .gitkeep
├── service/                  # 비즈니스 로직
│   ├── llm_service.py       # LLM 서비스
│   ├── embedding_service.py # Embedding 서비스
│   └── rag_service.py       # RAG 서비스
├── repository/              # 데이터 접근 계층
│   └── vector_store.py      # 벡터 스토어 리포지토리
├── router/                  # API 엔드포인트
│   ├── chat.py              # 챗봇 API
│   └── health.py            # 헬스 체크 API
├── config/                  # 설정 관리
│   └── settings.py          # 애플리케이션 설정
├── dependencies.py          # FastAPI 의존성 주입
├── api_server.py            # FastAPI 앱 (기존)
└── main.py                  # 진입점
```

## 모델 주입 방법

### 1. 모델 구현체 생성

`api/app/model/implementations/` 디렉토리에 모델 구현을 추가합니다:

```python
# api/app/model/implementations/my_model.py
from ..base import BaseLLM
from langchain_core.language_models import BaseChatModel

class MyLLMModel(BaseLLM):
    def __init__(self, model_path: str):
        # 모델 초기화
        pass

    def get_langchain_model(self) -> BaseChatModel:
        # LangChain 호환 모델 반환
        pass

    def invoke(self, prompt: str, **kwargs) -> str:
        # 프롬프트 실행
        pass

    # ... 나머지 메서드 구현
```

### 2. 모델 등록

`api_server.py`의 `startup_event`에서 모델을 등록합니다:

```python
from app.model.factory import LLMFactory, EmbeddingFactory
from app.model.implementations.my_model import MyLLMModel

@app.on_event("startup")
async def startup_event():
    # 모델 등록
    my_model = MyLLMModel(model_path="path/to/model")
    LLMFactory.register("my_model", my_model, is_default=True)

    # 나머지 초기화...
```

### 3. 환경 변수로 모델 선택

`.env` 파일에서 기본 모델을 설정:

```env
DEFAULT_LLM_MODEL=my_model
DEFAULT_EMBEDDING_MODEL=my_embedding
```

## 아키텍처 특징

- **의존성 주입**: FastAPI의 의존성 주입 시스템 활용
- **팩토리 패턴**: 모델 등록 및 조회를 위한 팩토리 패턴
- **인터페이스 분리**: BaseLLM, BaseEmbedding 인터페이스로 추상화
- **계층 분리**: Model → Service → Repository → Router 계층 구조
- **확장성**: 새로운 모델 추가 시 구현체만 추가하면 됨

## 사용 예시

```python
# 모델 등록
from app.model.factory import LLMFactory
from app.model.implementations.custom_model import CustomModel

custom_model = CustomModel()
LLMFactory.register("custom", custom_model, is_default=True)

# 서비스에서 사용
from app.service.llm_service import LLMService

llm_service = LLMService()  # 기본 모델 사용
# 또는
llm_service = LLMService(model_name="custom")  # 특정 모델 사용
```

