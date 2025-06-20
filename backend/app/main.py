from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .rag_chain import stream_answer
from typing import Optional


app = FastAPI(title="AI RAG Chatbot")

# CORS for your front-end origin on Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    history_summary: Optional[str] = None   # ← make it optional/ignored

@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    generator = stream_answer(req.question)
    return StreamingResponse(generator, media_type="text/plain")

