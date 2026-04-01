"""FastAPI layer for the Data Science Agent.

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import math
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import DEFAULT_MODEL, AgentSession

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    from langfuse import get_client

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    LangfuseCallbackHandler = None
    get_client = None


# ---------------------------------------------------------------------------
# JSON helpers — DataFrames can contain NaN / inf which stdlib json rejects.
# ---------------------------------------------------------------------------
def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN / inf floats with None for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def SafeJSONResponse(content: Any, **kwargs) -> JSONResponse:
    """JSONResponse that converts NaN/inf → null so it never crashes."""
    return JSONResponse(content=_sanitize(content), **kwargs)


# ---------------------------------------------------------------------------
# In-memory session store
# Replace with Redis / DB-backed store for production multi-worker deploys.
# ---------------------------------------------------------------------------
_sessions: dict[str, AgentSession] = {}
_langfuse_client = None
_langfuse_handler = None


def _get_session(session_id: str) -> AgentSession:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def _init_langfuse():
    global _langfuse_client, _langfuse_handler
    if not LANGFUSE_AVAILABLE:
        return
    try:
        from langfuse import get_client

        _langfuse_client = get_client()
        _langfuse_handler = LangfuseCallbackHandler()
    except Exception:
        pass


def _parse_metadata_fields(include_metadata: str | None) -> set[str]:
    if not include_metadata:
        return set()

    fields = {
        item.strip().lower() for item in include_metadata.split(",") if item.strip()
    }
    allowed_fields = {"session", "messages", "figures", "tool-results", "all"}
    invalid_fields = fields - allowed_fields
    if invalid_fields:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid X-Include-Metadata value(s): "
                + ", ".join(sorted(invalid_fields))
            ),
        )

    if "all" in fields:
        return {"session", "messages", "figures", "tool-results"}

    return fields


def _build_query_metadata(
    session: AgentSession, include_fields: set[str]
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    if "session" in include_fields:
        metadata["session"] = {
            "thread_id": session.thread_id,
            "model": session.model,
            "has_data": session.df is not None,
            "has_llm": session.llm_with_tools is not None,
            "message_count": len(session.messages),
            "figure_count": len(session.figures),
            "uploaded_file": session.uploaded_file_signature,
        }
    if "messages" in include_fields:
        metadata["messages"] = session.messages
    if "figures" in include_fields:
        metadata["figures"] = session.figures
    if "tool-results" in include_fields:
        metadata["tool_results"] = session.last_tool_results

    return metadata


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize Langfuse if available
    _init_langfuse()
    yield
    # Shutdown: clean up sessions
    _sessions.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Data Science Agent API",
    description="RESTful API for the Data Science Agent — upload data, ask questions, get AI-powered insights.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model name to use.",
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    model: str


class SetApiKeyRequest(BaseModel):
    api_key: str


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    figures: Optional[list[dict[str, Any]]] = None
    metadata: Optional[dict[str, Any]] = None


class SessionInfoResponse(BaseModel):
    session_id: str
    model: str
    has_data: bool
    has_llm: bool
    message_count: int
    figure_count: int
    uploaded_file: Optional[dict] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# -- Session lifecycle -------------------------------------------------------
@app.post("/sessions", response_model=CreateSessionResponse, tags=["Sessions"])
def create_session(body: CreateSessionRequest):
    """Create a new agent session."""
    session = AgentSession(model=body.model)
    session_id = uuid.uuid4().hex
    _sessions[session_id] = session
    return CreateSessionResponse(
        session_id=session_id,
        model=session.model,
    )


@app.get("/sessions", tags=["Sessions"])
def list_sessions():
    """List all active sessions."""
    return [
        {
            "session_id": sid,
            "model": s.model,
            "has_data": s.df is not None,
            "message_count": len(s.messages),
        }
        for sid, s in _sessions.items()
    ]


@app.get(
    "/sessions/{session_id}", response_model=SessionInfoResponse, tags=["Sessions"]
)
def get_session_info(session_id: str):
    """Get detailed info about a session."""
    s = _get_session(session_id)
    return SessionInfoResponse(
        session_id=session_id,
        model=s.model,
        has_data=s.df is not None,
        has_llm=s.llm_with_tools is not None,
        message_count=len(s.messages),
        figure_count=len(s.figures),
        uploaded_file=s.uploaded_file_signature,
    )


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(session_id: str):
    """Delete (destroy) a session entirely."""
    _get_session(session_id)  # ensure it exists
    del _sessions[session_id]
    return {"detail": "Session deleted"}


@app.post("/sessions/{session_id}/clear", tags=["Sessions"])
def clear_session(session_id: str):
    """Clear chat history and memory but keep the session alive."""
    s = _get_session(session_id)
    s.clear_memory()
    return {"detail": "Session memory cleared"}


# -- API key -----------------------------------------------------------------
@app.post("/sessions/{session_id}/api-key", tags=["Configuration"])
def set_api_key(session_id: str, body: SetApiKeyRequest):
    """Set or update the Google API key for a session."""
    s = _get_session(session_id)
    try:
        s.set_api_key(body.api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"detail": "API key set", "model": s.model}


# -- File upload -------------------------------------------------------------
@app.post("/sessions/{session_id}/upload", tags=["Data"])
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload a CSV / Excel file to a session."""
    s = _get_session(session_id)
    file_bytes = await file.read()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    try:
        s.load_uploaded_file(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "filename": file.filename,
        "shape": list(s.df.shape) if s.df is not None else None,
    }


@app.get("/sessions/{session_id}/data/preview", tags=["Data"])
def preview_data(session_id: str, rows: int = 5):
    """Preview the first N rows of the uploaded dataset."""
    s = _get_session(session_id)
    if s.df is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet")
    return SafeJSONResponse(
        {
            "shape": list(s.df.shape),
            "columns": list(s.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in s.df.dtypes.items()},
            "preview": s.df.head(rows).to_dict(orient="records"),
        }
    )


# -- Chat / Query ------------------------------------------------------------
@app.post("/sessions/{session_id}/query", response_model=QueryResponse, tags=["Chat"])
def run_query(
    session_id: str,
    body: QueryRequest,
    x_include_metadata: str | None = Header(default=None),
):
    """Send a natural-language query to the agent."""
    s = _get_session(session_id)
    include_fields = _parse_metadata_fields(x_include_metadata)

    if s.df is None:
        raise HTTPException(status_code=400, detail="Upload data before querying")
    if s.graph is None:
        raise HTTPException(
            status_code=400,
            detail="API key not set — call POST /sessions/{id}/api-key first",
        )

    try:
        result = s.run(
            query=body.query,
            langfuse_handler=_langfuse_handler,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = {
        "answer": result.get("answer", ""),
    }
    if result.get("figures"):
        response["figures"] = result["figures"]

    metadata = _build_query_metadata(s, include_fields)
    if metadata:
        response["metadata"] = metadata

    return SafeJSONResponse(response)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health():
    langfuse_status = "available" if LANGFUSE_AVAILABLE else "not available"
    if LANGFUSE_AVAILABLE and _langfuse_handler is not None:
        langfuse_status = "initialized"
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
        "langfuse": langfuse_status,
    }
