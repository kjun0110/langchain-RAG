"use client";

import { useState, useRef, useEffect } from "react";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

type ModelType = "openai" | "local";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "안녕하세요! LangChain 챗봇입니다. 무엇을 도와드릴까요?",
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [modelType, setModelType] = useState<ModelType>("openai");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // 디버깅: 전송하는 model_type 확인
      console.log("[DEBUG] 전송하는 model_type:", modelType);

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: content,
          history: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          model_type: modelType, // "openai" 또는 "local"
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.error || errorData.message || "응답을 받는 중 오류가 발생했습니다.";

        // OpenAI 호출량 초과 에러
        if (response.status === 429) {
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: "⚠️ OpenAI API 호출량이 초과되었습니다. 할당량을 확인하고 다시 시도해주세요. 또는 '로컬 모델' 버튼을 선택하여 로컬 모델을 사용해주세요.",
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errorMessage]);
          return;
        }

        // 백엔드 연결 오류
        if (response.status === 503) {
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: `⚠️ ${errorMsg}`,
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errorMessage]);
          return;
        }

        throw new Error(errorMsg);
      }

      const data = await response.json();

      // 에러 응답인지 확인
      if (data.error) {
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: `⚠️ ${data.error}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
        return;
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response || "응답을 생성할 수 없습니다.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `⚠️ 오류가 발생했습니다: ${error instanceof Error ? error.message : "알 수 없는 오류"}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>🤖 LangChain Chatbot</h1>
        <p>PGVector와 연동된 AI 챗봇</p>
        <div className="model-selector">
          <button
            className={`model-button ${modelType === "openai" ? "active" : ""}`}
            onClick={() => setModelType("openai")}
            disabled={isLoading}
          >
            🌐 OpenAI
          </button>
          <button
            className={`model-button ${modelType === "local" ? "active" : ""}`}
            onClick={() => setModelType("local")}
            disabled={isLoading}
          >
            🖥️ 로컬 모델
          </button>
        </div>
      </header>

      <main className="chat-main">
        <div className="messages-container">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isLoading && (
            <div className="loading-indicator">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <ChatInput onSend={handleSendMessage} disabled={isLoading} />
      </main>

      <style jsx>{`
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100vh;
          max-width: 800px;
          margin: 0 auto;
          background: white;
          box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }

        .chat-header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 1.5rem;
          text-align: center;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .chat-header h1 {
          font-size: 1.5rem;
          margin-bottom: 0.5rem;
        }

        .chat-header p {
          font-size: 0.9rem;
          opacity: 0.9;
        }

        .model-selector {
          display: flex;
          gap: 0.5rem;
          margin-top: 1rem;
          justify-content: center;
        }

        .model-button {
          padding: 0.5rem 1rem;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-radius: 0.5rem;
          background: rgba(255, 255, 255, 0.1);
          color: white;
          font-size: 0.9rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .model-button:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.2);
          border-color: rgba(255, 255, 255, 0.5);
        }

        .model-button.active {
          background: rgba(255, 255, 255, 0.3);
          border-color: white;
          font-weight: 600;
        }

        .model-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .chat-main {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .messages-container {
          flex: 1;
          overflow-y: auto;
          padding: 1rem;
          background: #f5f5f5;
        }

        .loading-indicator {
          display: flex;
          justify-content: flex-start;
          padding: 1rem;
        }

        .typing-dots {
          display: flex;
          gap: 0.5rem;
          padding: 1rem 1.5rem;
          background: white;
          border-radius: 1.5rem;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        .typing-dots span {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #667eea;
          animation: typing 1.4s infinite;
        }

        .typing-dots span:nth-child(2) {
          animation-delay: 0.2s;
        }

        .typing-dots span:nth-child(3) {
          animation-delay: 0.4s;
        }

        @keyframes typing {
          0%,
          60%,
          100% {
            transform: translateY(0);
            opacity: 0.7;
          }
          30% {
            transform: translateY(-10px);
            opacity: 1;
          }
        }

        @media (max-width: 768px) {
          .chat-container {
            height: 100vh;
            height: 100dvh; /* 모바일 브라우저 높이 */
          }

          .chat-header h1 {
            font-size: 1.2rem;
          }
        }
      `}</style>
    </div>
  );
}

