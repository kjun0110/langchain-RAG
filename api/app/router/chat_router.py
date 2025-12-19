"""
FastAPI 기준의 API 엔드포인트 계층입니다.

chat_router.py
POST /api/chat
세션 ID, 메시지 리스트 등을 받아 대화형 응답 반환.
"""

import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """챗봇 요청 모델."""

    message: str
    history: Optional[List[dict]] = []
    model_type: Optional[str] = "openai"  # "openai" 또는 "local"


class ChatResponse(BaseModel):
    """챗봇 응답 모델."""

    response: str


def get_chat_service():
    """ChatService 인스턴스를 반환하는 함수.

    이 함수는 api_server.py의 전역 변수에 접근하기 위해
    api_server 모듈에서 import하여 사용합니다.
    순환 import 방지를 위해 함수 내부에서 import합니다.
    """
    # 순환 import 방지를 위해 함수 내부에서 import
    import sys

    # api_server 모듈이 이미 로드되어 있는지 확인
    if "app.api_server" in sys.modules:
        from .. import api_server
    else:
        # 모듈이 아직 로드되지 않은 경우 직접 import
        import importlib

        api_server = importlib.import_module("app.api_server")

    return api_server.chat_service


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """챗봇 API 엔드포인트 - ChatService를 사용한 RAG 체인."""
    # ChatService 인스턴스 가져오기
    chat_service = get_chat_service()
    if chat_service is None:
        raise HTTPException(
            status_code=503,
            detail="ChatService가 초기화되지 않았습니다. 서버를 재시작해주세요.",
        )

    # 요청자의 IP 주소 확인 (localhost 여부 판단)
    client_host = http_request.client.host if http_request.client else None
    is_localhost = (
        client_host == "127.0.0.1"
        or client_host == "localhost"
        or client_host == "::1"
        or (client_host and client_host.startswith("127."))
    )

    # 모델 타입에 따라 적절한 RAG 체인 선택
    # 프론트엔드에서 전달된 model_type이 없으면 .env의 LLM_PROVIDER 사용
    model_type = request.model_type or os.getenv("LLM_PROVIDER", "openai")
    if model_type:
        model_type = model_type.lower()

    # 환경과 모델 타입 불일치 검증
    if is_localhost and model_type == "openai":
        raise HTTPException(
            status_code=400,
            detail="현재 로컬환경입니다. 로컬 모델을 사용해주세요.",
        )

    if not is_localhost and model_type == "local":
        raise HTTPException(
            status_code=400,
            detail="현재 로컬 환경이 아닙니다. OpenAI 모델을 사용해주세요.",
        )

    # 디버깅: 받은 model_type 로그 출력
    print(
        f"[DEBUG] 받은 model_type: {request.model_type}, 처리된 model_type: {model_type}, client_host: {client_host}, is_localhost: {is_localhost}"
    )

    try:
        # ChatService를 통해 RAG 체인 실행
        response_text = chat_service.chat_with_rag(
            message=request.message,
            history=request.history,
            model_type=model_type,
        )

        return ChatResponse(response=response_text)

    except RuntimeError as e:
        error_msg = str(e)
        print(f"[ERROR] 챗봇 응답 생성 실패: {error_msg}")

        # OpenAI API 할당량 초과 에러 확인
        if "할당량" in error_msg or "quota" in error_msg.lower():
            error_detail = (
                "⚠️ OpenAI API 할당량이 초과되었습니다.\n\n"
                "해결 방법:\n"
                "1. OpenAI 계정의 사용량 및 할당량을 확인하세요\n"
                "2. OpenAI 계정에 결제 정보를 추가하거나 할당량을 늘리세요\n"
                "3. 또는 '🖥️ 로컬 모델' 버튼을 선택하여 로컬 Midm 모델을 사용하세요"
            )
            raise HTTPException(
                status_code=503,
                detail=error_detail,
            )
        else:
            raise HTTPException(
                status_code=503,
                detail=error_msg,
            )

    except ValueError as e:
        error_msg = str(e)
        print(f"[ERROR] 잘못된 요청: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=error_msg,
        )

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] 챗봇 응답 생성 실패: {error_msg}")

        # OpenAI API 호출량 초과 에러 확인
        if (
            "quota" in error_msg.lower()
            or "429" in error_msg
            or "insufficient_quota" in error_msg
            or "exceeded" in error_msg.lower()
        ):
            error_detail = "OpenAI API 호출량이 초과되었습니다. 할당량을 확인하고 다시 시도해주세요."
            raise HTTPException(
                status_code=429,
                detail=error_detail,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"응답 생성 중 오류가 발생했습니다: {error_msg[:200]}",
            )
