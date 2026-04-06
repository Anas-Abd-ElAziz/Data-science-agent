"""FastAPI layer for the Data Science Agent.

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import (
    DEFAULT_MODEL,
    AgentSession,
    DatasetNotFoundError,
    DatasetStoreConfigError,
    DatasetStoreError,
    S3DatasetStore,
    build_redis_checkpointer,
)

from agent.session_store import RedisSessionStore

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    LangfuseCallbackHandler = None


logger = logging.getLogger(__name__)


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


def safe_json_response(content: Any, **kwargs) -> JSONResponse:
    """JSONResponse that converts NaN/inf → null so it never crashes."""
    return JSONResponse(content=_sanitize(content), **kwargs)


# ---------------------------------------------------------------------------
# Live in-memory session runtime
# ---------------------------------------------------------------------------
_sessions: dict[str, AgentSession] = {}
_langfuse_handler = None
_session_store: RedisSessionStore | None = None
_graph_checkpointer = None
_dataset_store: S3DatasetStore | None = None


def _serialize_recent_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
            "timestamp": message.get("timestamp"),
            "figure_count": len(message.get("figures", []) or []),
            "figures": message.get("figures", []) or [],
        }
        for message in messages[-20:]
    ]


def _collect_message_figures(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    for message in messages:
        figures.extend(message.get("figures", []) or [])
    return figures


def _build_session_metadata(session_id: str, session: AgentSession) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "model": session.model,
        "thread_id": session.thread_id,
        "has_data": session.has_data(),
        "message_count": len(session.messages),
        "figure_count": len(session.figures),
        "uploaded_file": session.uploaded_file_signature,
    }


def _sync_session_store(session_id: str, session: AgentSession) -> None:
    if _session_store is None:
        return

    try:
        _session_store.upsert_session_meta(
            session_id, _build_session_metadata(session_id, session)
        )
        _session_store.replace_messages(
            session_id, _serialize_recent_messages(session.messages)
        )
        _session_store.replace_figures(session_id, session.figures)
    except Exception:
        logger.exception(
            "Failed to sync session '%s' to Redis session store.", session_id
        )


def _restore_session_from_store(session_id: str) -> AgentSession | None:
    if _session_store is None:
        return None

    try:
        metadata = _session_store.get_session_meta(session_id)
        if not metadata:
            return None

        session = AgentSession(
            model=metadata.get("model", DEFAULT_MODEL),
            checkpointer=_graph_checkpointer,
            dataset_store=_dataset_store,
            session_id=session_id,
        )
        session.thread_id = metadata.get("thread_id") or session.thread_id
        session.uploaded_file_signature = metadata.get("uploaded_file")
        session.dataset_ref = (session.uploaded_file_signature or {}).get("dataset")
        if session.uploaded_file_signature:
            session.file_signature = {
                "name": session.uploaded_file_signature.get("name"),
                "size": session.uploaded_file_signature.get("size"),
                "sha256": session.uploaded_file_signature.get("sha256"),
            }
        session.messages = _session_store.get_messages(session_id)
        session.figures = _session_store.get_figures(
            session_id
        ) or _collect_message_figures(session.messages)
    except Exception:
        logger.exception("Failed to restore session '%s' from Redis.", session_id)
        return None

    _sessions[session_id] = session
    return session


def _delete_session_store_state(session_id: str) -> None:
    if _session_store is not None:
        try:
            _session_store.delete_session(session_id)
        except Exception:
            logger.exception(
                "Failed to delete session '%s' from Redis session store.", session_id
            )


def _raise_dataset_http_error(exc: Exception) -> None:
    if isinstance(exc, DatasetNotFoundError):
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    if isinstance(exc, DatasetStoreConfigError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if isinstance(exc, DatasetStoreError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


def _get_session(session_id: str) -> AgentSession:
    session = _sessions.get(session_id)
    if session is None:
        session = _restore_session_from_store(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def _redis_is_configured() -> bool:
    return bool(
        os.getenv("REDIS_URL") or (os.getenv("REDIS_HOST") and os.getenv("REDIS_PORT"))
    )


def _s3_is_configured() -> bool:
    return bool(os.getenv("S3_DATASET_BUCKET"))


def _init_langfuse():
    global _langfuse_handler
    if not LANGFUSE_AVAILABLE:
        return
    try:
        _langfuse_handler = LangfuseCallbackHandler()
    except Exception:
        logger.exception("Failed to initialize Langfuse.")


def _init_session_store():
    global _session_store
    try:
        _session_store = RedisSessionStore.from_env()
    except Exception:
        _session_store = None
        logger.exception("Failed to initialize Redis session store.")
        return

    if _session_store is None and _redis_is_configured():
        logger.warning(
            "Redis was configured, but the session store did not initialize."
        )


def _init_graph_checkpointer():
    global _graph_checkpointer
    try:
        _graph_checkpointer = build_redis_checkpointer()
    except Exception:
        _graph_checkpointer = None
        logger.exception("Failed to initialize Redis graph checkpointer.")
        return

    if _graph_checkpointer is None and _redis_is_configured():
        logger.warning(
            "Redis was configured, but the graph checkpointer did not initialize."
        )


def _init_dataset_store():
    global _dataset_store
    try:
        _dataset_store = S3DatasetStore.from_env()
    except Exception:
        _dataset_store = None
        logger.exception("Failed to initialize S3 dataset store.")
        return

    if _dataset_store is None and _s3_is_configured():
        logger.warning(
            "S3 dataset storage was configured, but the dataset store did not initialize."
        )


def _parse_metadata_fields(include_metadata: str | None) -> set[str]:
    if not include_metadata:
        return set()

    fields = {
        item.strip().lower().replace("_", "-")
        for item in include_metadata.split(",")
        if item.strip()
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
            "has_data": session.has_data(),
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
    _init_session_store()
    _init_graph_checkpointer()
    _init_dataset_store()
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
    allow_credentials=False,
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
    session = AgentSession(
        model=body.model,
        checkpointer=_graph_checkpointer,
        dataset_store=_dataset_store,
    )
    session_id = uuid.uuid4().hex
    session.set_session_id(session_id)
    _sessions[session_id] = session
    _sync_session_store(session_id, session)
    return CreateSessionResponse(
        session_id=session_id,
        model=session.model,
    )


@app.get("/sessions", tags=["Sessions"])
def list_sessions():
    """List all active sessions."""
    if _session_store is not None:
        try:
            stored_sessions = _session_store.list_sessions()
        except Exception:
            logger.exception("Failed to list Redis-backed sessions.")
            stored_sessions = None

        if stored_sessions is not None:
            return [
                {
                    "session_id": item["session_id"],
                    "model": item.get("model", DEFAULT_MODEL),
                    "has_data": bool(item.get("has_data")),
                    "has_llm": bool(
                        _sessions.get(item["session_id"])
                        and _sessions[item["session_id"]].llm_with_tools is not None
                    ),
                    "message_count": int(item.get("message_count", 0)),
                }
                for item in stored_sessions
            ]

    return [
        {
            "session_id": sid,
            "model": s.model,
            "has_data": s.has_data(),
            "has_llm": s.llm_with_tools is not None,
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
        has_data=s.has_data(),
        has_llm=s.llm_with_tools is not None,
        message_count=len(s.messages),
        figure_count=len(s.figures),
        uploaded_file=s.uploaded_file_signature,
    )


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(session_id: str):
    """Delete (destroy) a session entirely."""
    session = _get_session(session_id)
    session.delete_uploaded_dataset()
    session.close()
    del _sessions[session_id]
    _delete_session_store_state(session_id)
    return {"detail": "Session deleted"}


@app.post("/sessions/{session_id}/clear", tags=["Sessions"])
def clear_session(session_id: str):
    """Clear chat history and memory but keep the session alive."""
    s = _get_session(session_id)
    s.clear_memory()
    if _session_store is not None:
        try:
            _session_store.clear_session(session_id, s.thread_id)
        except Exception:
            logger.exception(
                "Failed to clear session '%s' in Redis session store.", session_id
            )
    return {"detail": "Session memory cleared"}


# -- API key -----------------------------------------------------------------
@app.post("/sessions/{session_id}/api-key", tags=["Configuration"])
def set_api_key(session_id: str, body: SetApiKeyRequest):
    """Set or update the Gemini API key for a session."""
    s = _get_session(session_id)
    try:
        s.set_api_key(body.api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    _sync_session_store(session_id, s)
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
        s.load_uploaded_file(file_bytes, file.filename, session_id=session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (DatasetNotFoundError, DatasetStoreConfigError, DatasetStoreError) as e:
        _raise_dataset_http_error(e)

    _sync_session_store(session_id, s)

    return {
        "filename": file.filename,
        "shape": list(s.get_df().shape),
    }


@app.get("/sessions/{session_id}/data/preview", tags=["Data"])
def preview_data(session_id: str, rows: int = Query(default=5, ge=1, le=100)):
    """Preview the first N rows of the uploaded dataset."""
    s = _get_session(session_id)
    if not s.has_data():
        raise HTTPException(status_code=400, detail="No data uploaded yet")

    try:
        df = s.get_df()
    except (DatasetNotFoundError, DatasetStoreConfigError, DatasetStoreError) as e:
        _raise_dataset_http_error(e)

    return safe_json_response(
        {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(rows).to_dict(orient="records"),
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

    if not s.has_data():
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
    except (DatasetNotFoundError, DatasetStoreConfigError, DatasetStoreError) as e:
        _raise_dataset_http_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _sync_session_store(session_id, s)

    response = {
        "answer": result.get("answer", ""),
    }
    if result.get("figures"):
        response["figures"] = result["figures"]

    metadata = _build_query_metadata(s, include_fields)
    if metadata:
        response["metadata"] = metadata

    return safe_json_response(response)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health():
    from agent.helpers import NSJAIL_AVAILABLE

    degraded = False
    langfuse_status = "available" if LANGFUSE_AVAILABLE else "not available"
    if LANGFUSE_AVAILABLE and _langfuse_handler is not None:
        langfuse_status = "initialized"
    graph_checkpointer_status = "memory"
    if _graph_checkpointer is not None:
        graph_checkpointer_status = "redis"

    dataset_store_status = "disabled"
    if _dataset_store is not None:
        try:
            dataset_store_status = (
                "connected" if _dataset_store.ping() else "unavailable"
            )
        except Exception:
            logger.exception("Failed to probe S3 dataset store health.")
            dataset_store_status = "error"
        degraded = dataset_store_status != "connected"

    session_store_status = "disabled"
    active_sessions = len(_sessions)
    if _session_store is not None:
        try:
            session_store_status = (
                "connected" if _session_store.ping() else "unavailable"
            )
        except Exception:
            logger.exception("Failed to probe Redis session store health.")
            session_store_status = "error"

        try:
            active_sessions = len(_session_store.list_sessions())
        except Exception:
            logger.exception(
                "Failed to count Redis-backed sessions during health check."
            )
            active_sessions = len(_sessions)
            if session_store_status == "connected":
                session_store_status = "error"

        degraded = degraded or session_store_status != "connected"

    return {
        "status": "degraded" if degraded else "ok",
        "active_sessions": active_sessions,
        "dataset_store": dataset_store_status,
        "graph_checkpointer": graph_checkpointer_status,
        "langfuse": langfuse_status,
        "python_repl_backend": "nsjail+subprocess"
        if NSJAIL_AVAILABLE
        else "subprocess",
        "session_store": session_store_status,
    }
