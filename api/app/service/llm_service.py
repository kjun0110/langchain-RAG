"""LLM 서비스 - LLM 모델 사용."""

from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from ..model.base import BaseLLM
from ..model.factory import LLMFactory


class LLMService:
    """LLM 서비스."""

    def __init__(self, model_name: Optional[str] = None):
        """LLM 서비스 초기화."""
        self.llm: BaseLLM = LLMFactory.get(model_name)

    def generate(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        """텍스트 생성."""
        return self.llm.invoke(prompt, **kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        """스트리밍 텍스트 생성."""
        return self.llm.stream(prompt, **kwargs)

    def get_langchain_model(self):
        """LangChain 호환 모델 반환."""
        return self.llm.get_langchain_model()

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환."""
        return self.llm.get_model_info()

