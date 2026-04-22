import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from moroccan_legal_chatbot import AdvancedLegalChatbot, Config, logger


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="web_session")


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    service: str


app = FastAPI(
    title="Moroccan Legal Assistant API",
    description="API backend for React frontend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot: Optional[AdvancedLegalChatbot] = None


@app.on_event("startup")
def startup_event() -> None:
    # Reduces tokenizer threading overhead in constrained machines.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logger.info("Legal API server started; chatbot will initialize on first request")


def get_chatbot() -> AdvancedLegalChatbot:
    global chatbot
    if chatbot is None:
        chatbot = AdvancedLegalChatbot()
        logger.info("Legal chatbot initialized on-demand")
    return chatbot


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="moroccan-legal-assistant-api")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        assistant = get_chatbot()
        answer = assistant.chat(payload.message, payload.session_id)
        return ChatResponse(answer=answer, session_id=payload.session_id)
    except Exception as exc:
        logger.error("Chat endpoint failed: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Chat request failed")


@app.post("/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    session_id: str = Form(default="web_session"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = Path(Config.USER_UPLOADS_PATH)
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = upload_dir / temp_filename

    try:
        assistant = get_chatbot()
        content = await file.read()
        temp_path.write_bytes(content)

        analysis = assistant.doc_analyzer.analyze_document(str(temp_path))
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])

        return {
            "session_id": session_id,
            "filename": file.filename,
            "metadata": analysis.get("metadata", {}),
            "summary": analysis.get("summary", ""),
            "related_laws": analysis.get("related_laws", []),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document analysis endpoint failed: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Document analysis failed")
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            logger.warning("Could not delete temporary uploaded file: %s", str(temp_path))

