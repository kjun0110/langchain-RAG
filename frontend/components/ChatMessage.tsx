interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="message-content">
        <div className="message-text">{message.content}</div>
        <div className="message-time">
          {message.timestamp.toLocaleTimeString("ko-KR", {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </div>
      </div>

      <style jsx>{`
        .message {
          display: flex;
          margin-bottom: 1rem;
          animation: fadeIn 0.3s ease-in;
        }

        .message.user {
          justify-content: flex-end;
        }

        .message.assistant {
          justify-content: flex-start;
        }

        .message-content {
          max-width: 70%;
          padding: 0.75rem 1rem;
          border-radius: 1.5rem;
          word-wrap: break-word;
        }

        .message.user .message-content {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-bottom-right-radius: 0.25rem;
        }

        .message.assistant .message-content {
          background: white;
          color: #333;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
          border-bottom-left-radius: 0.25rem;
        }

        .message-text {
          line-height: 1.5;
          white-space: pre-wrap;
        }

        .message-time {
          font-size: 0.75rem;
          opacity: 0.7;
          margin-top: 0.25rem;
          text-align: right;
        }

        .message.assistant .message-time {
          text-align: left;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @media (max-width: 768px) {
          .message-content {
            max-width: 85%;
          }
        }
      `}</style>
    </div>
  );
}

