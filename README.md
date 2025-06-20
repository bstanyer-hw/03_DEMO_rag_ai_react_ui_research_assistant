# AI RAG Chatbot

A full-stack template for a public-facing, multi-agent LangChain chatbot.

## 🏗  Stack
- **FastAPI** + LangChain (Python) — token-streaming back-end
- **React 18** + Tailwind CSS — professional front-end
- **Render** & **Vercel** — one-click free deployment

## 🚀  Quick start

```bash
git clone https://github.com/your-org/ai-rag-chatbot.git
cd ai-rag-chatbot

# Back-end
cp .env.example .env              # fill in keys
docker compose up --build backend # or `uvicorn` locally

# Front-end
cd frontend
npm i
npm run dev                       # http://localhost:5173
