"""챗봇 API 라우터."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..service.rag_service import RAGService
from ..dependencies import get_rag_service

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """챗봇 요청 모델."""

    message: str
    history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    """챗봇 응답 모델."""

    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """챗봇 API 엔드포인트."""
    try:
        response_text = rag_service.generate(request.message, request.history)
        return ChatResponse(response=response_text)
    except Exception as e:
        error_msg = str(e)
        if (
            "quota" in error_msg.lower()
            or "429" in error_msg
            or "insufficient_quota" in error_msg
            or "exceeded" in error_msg.lower()
        ):
            raise HTTPException(
                status_code=429,
                detail="OpenAI API 호출량이 초과되었습니다. 할당량을 확인하고 다시 시도해주세요.",
            )
        raise HTTPException(
            status_code=500,
            detail=f"응답 생성 중 오류가 발생했습니다: {error_msg[:200]}",
        )

