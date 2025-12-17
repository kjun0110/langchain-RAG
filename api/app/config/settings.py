"""애플리케이션 설정."""

import os
from typing import Optional


class Settings:
    """애플리케이션 설정."""

    # 데이터베이스 설정
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: str = os.getenv("POSTGRES_PORT", "5432")
    postgres_user: str = os.getenv("POSTGRES_USER", "langchain")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "langchain123")
    postgres_db: str = os.getenv("POSTGRES_DB", "langchain_db")

    # 모델 설정
    default_llm_model: Optional[str] = os.getenv("DEFAULT_LLM_MODEL", None)
    default_embedding_model: Optional[str] = os.getenv("DEFAULT_EMBEDDING_MODEL", None)

    # OpenAI 설정
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)

    @property
    def connection_string(self) -> str:
        """PostgreSQL 연결 문자열."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def collection_name(self) -> str:
        """벡터 컬렉션 이름."""
        return "langchain_collection"



_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """설정 싱글톤 반환."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

