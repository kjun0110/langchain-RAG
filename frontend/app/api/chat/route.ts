import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { message, history, model_type } = body;

    if (!message) {
      return NextResponse.json(
        { error: "메시지가 필요합니다." },
        { status: 400 }
      );
    }

    // LangChain 백엔드 API 호출
    // docker-compose의 langchain-app 서비스와 통신
    const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";

    try {
      // 디버깅: 받은 model_type 확인
      console.log("[DEBUG] Next.js API route에서 받은 model_type:", model_type);

      const response = await fetch(`${backendUrl}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          history: history || [],
          model_type: model_type, // 프론트엔드에서 전달된 값 그대로 사용 (기본값 없음)
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || errorData.message || response.statusText;

        // OpenAI 호출량 초과 에러 (429)
        if (response.status === 429) {
          return NextResponse.json(
            {
              error: "OpenAI API 호출량이 초과되었습니다. 할당량을 확인하고 다시 시도해주세요.",
              status: 429,
            },
            { status: 429 }
          );
        }

        // 기타 백엔드 오류
        return NextResponse.json(
          {
            error: errorMessage || "백엔드 서버 오류가 발생했습니다.",
            status: response.status,
          },
          { status: response.status }
        );
      }

      const data = await response.json();
      return NextResponse.json({ response: data.response || data.message });
    } catch (backendError) {
      // 백엔드 연결 실패 (서버가 실행되지 않음)
      console.error("Backend connection error:", backendError);

      // 네트워크 오류인지 확인
      if (backendError instanceof TypeError && backendError.message.includes("fetch")) {
        return NextResponse.json(
          {
            error: "백엔드 서버에 연결할 수 없습니다. 백엔드 서비스가 실행 중인지 확인해주세요.",
            status: 503,
          },
          { status: 503 }
        );
      }

      return NextResponse.json(
        {
          error: "백엔드 서버와 통신 중 오류가 발생했습니다.",
          status: 500,
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error("API error:", error);
    return NextResponse.json(
      {
        error: "서버 오류가 발생했습니다.",
        details: error instanceof Error ? error.message : "알 수 없는 오류",
      },
      { status: 500 }
    );
  }
}

