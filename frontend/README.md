# LangChain Chatbot - Next.js PWA

LangChain과 PGVector를 연동한 챗봇 서비스입니다.

## 기능

- 🤖 LangChain 기반 AI 챗봇
- 📱 PWA (Progressive Web App) 지원
- 💬 실시간 채팅 인터페이스
- 🔄 PGVector 벡터 검색 연동

## 설치 및 실행

### 1. 의존성 설치

```bash
npm install
```

### 2. 개발 서버 실행

```bash
npm run dev
```

브라우저에서 [http://localhost:3000](http://localhost:3000)을 열어 확인하세요.

### 3. 프로덕션 빌드

```bash
npm run build
npm start
```

## 환경 변수 설정

`.env.local` 파일을 생성하고 다음을 설정하세요:

```env
BACKEND_URL=http://localhost:8000
```

## PWA 설치

1. 모바일 브라우저에서 사이트 접속
2. 브라우저 메뉴에서 "홈 화면에 추가" 선택
3. 앱 아이콘이 홈 화면에 추가됩니다

## 백엔드 연동

LangChain 백엔드 서비스와 연동하려면:

1. `app.py`를 FastAPI나 Flask로 API 서버로 변환
2. `BACKEND_URL` 환경 변수 설정
3. `/api/chat` 엔드포인트 구현

## 프로젝트 구조

```
frontend/
├── app/
│   ├── api/chat/route.ts    # API 라우트
│   ├── layout.tsx            # 레이아웃
│   ├── page.tsx              # 메인 페이지
│   └── globals.css           # 전역 스타일
├── components/
│   ├── ChatMessage.tsx       # 메시지 컴포넌트
│   └── ChatInput.tsx         # 입력 컴포넌트
├── public/
│   └── manifest.json         # PWA 매니페스트
└── package.json
```

## 기술 스택

- **Next.js 14** - React 프레임워크
- **TypeScript** - 타입 안정성
- **next-pwa** - PWA 지원
- **LangChain** - AI 프레임워크 (백엔드)

