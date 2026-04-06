"""Redis-backed LangGraph checkpointer setup."""

from __future__ import annotations

import os
from urllib.parse import quote

try:
    from langgraph.checkpoint.redis import RedisSaver

    REDIS_CHECKPOINTER_AVAILABLE = True
except ImportError:
    RedisSaver = None
    REDIS_CHECKPOINTER_AVAILABLE = False


def _build_redis_url() -> str | None:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return redis_url

    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT")
    if not host or not port:
        return None

    username = quote(os.getenv("REDIS_USERNAME") or "default", safe="")
    password = os.getenv("REDIS_PASSWORD")
    auth_segment = f"{username}@"
    if password:
        auth_segment = f"{username}:{quote(password, safe='')}@"

    db = os.getenv("REDIS_DB", "0")
    return f"redis://{auth_segment}{host}:{port}/{db}"


def build_redis_checkpointer():
    """Create and initialize a shared Redis LangGraph checkpointer."""
    if not REDIS_CHECKPOINTER_AVAILABLE:
        return None

    redis_url = _build_redis_url()
    if not redis_url:
        return None

    saver = RedisSaver(redis_url=redis_url)
    saver.setup()
    return saver
