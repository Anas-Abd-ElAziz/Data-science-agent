"""Redis-backed durable session metadata and chat history."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import redis


DEFAULT_SESSION_TTL_SECONDS = 24 * 60 * 60
MAX_REDIS_MESSAGES = 20


class RedisSessionStore:
    """Stores session metadata and recent chat history in Redis."""

    def __init__(
        self, client: redis.Redis, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS
    ):
        self.client = client
        self.ttl_seconds = ttl_seconds

    @classmethod
    def from_env(cls) -> "RedisSessionStore | None":
        socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "3"))
        connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT_SECONDS", "3"))
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=connect_timeout,
            )
        else:
            host = os.getenv("REDIS_HOST")
            port = os.getenv("REDIS_PORT")
            if not host or not port:
                return None

            client = redis.Redis(
                host=host,
                port=int(port),
                decode_responses=True,
                username=os.getenv("REDIS_USERNAME") or None,
                password=os.getenv("REDIS_PASSWORD") or None,
                db=int(os.getenv("REDIS_DB", "0")),
                socket_timeout=socket_timeout,
                socket_connect_timeout=connect_timeout,
            )

        ttl_seconds = int(
            os.getenv("REDIS_SESSION_TTL_SECONDS", str(DEFAULT_SESSION_TTL_SECONDS))
        )
        return cls(client=client, ttl_seconds=max(60, ttl_seconds))

    @property
    def active_sessions_key(self) -> str:
        return "sessions:active"

    def _meta_key(self, session_id: str) -> str:
        return f"session:{session_id}:meta"

    def _messages_key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    def _figures_key(self, session_id: str) -> str:
        return f"session:{session_id}:figures"

    def ping(self) -> bool:
        return bool(self.client.ping())

    def get_session_meta(self, session_id: str) -> dict[str, Any]:
        raw = self.client.get(self._meta_key(session_id))
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def upsert_session_meta(self, session_id: str, metadata: dict[str, Any]) -> None:
        payload = dict(metadata)
        existing = self.get_session_meta(session_id)
        payload["session_id"] = session_id
        payload["created_at"] = (
            payload.get("created_at")
            or existing.get("created_at")
            or datetime.now(timezone.utc).isoformat()
        )
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.client.set(
            self._meta_key(session_id),
            json.dumps(payload),
            ex=self.ttl_seconds,
        )
        self.client.sadd(self.active_sessions_key, session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        session_ids = sorted(self.client.smembers(self.active_sessions_key))
        sessions: list[dict[str, Any]] = []
        stale_ids: list[str] = []
        for session_id in session_ids:
            metadata = self.get_session_meta(session_id)
            if metadata:
                sessions.append(metadata)
            else:
                stale_ids.append(session_id)

        if stale_ids:
            self.client.srem(self.active_sessions_key, *stale_ids)

        return sessions

    def replace_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        key = self._messages_key(session_id)
        pipeline = self.client.pipeline()
        pipeline.delete(key)
        trimmed_messages = messages[-MAX_REDIS_MESSAGES:]
        if trimmed_messages:
            pipeline.rpush(key, *[json.dumps(message) for message in trimmed_messages])
            pipeline.expire(key, self.ttl_seconds)
        pipeline.execute()

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for raw_message in self.client.lrange(self._messages_key(session_id), 0, -1):
            try:
                messages.append(json.loads(raw_message))
            except json.JSONDecodeError:
                continue
        return messages

    def replace_figures(self, session_id: str, figures: list[dict[str, Any]]) -> None:
        key = self._figures_key(session_id)
        if not figures:
            self.client.delete(key)
            return

        self.client.set(key, json.dumps(figures), ex=self.ttl_seconds)

    def get_figures(self, session_id: str) -> list[dict[str, Any]]:
        raw = self.client.get(self._figures_key(session_id))
        if not raw:
            return []

        try:
            figures = json.loads(raw)
        except json.JSONDecodeError:
            return []

        return figures if isinstance(figures, list) else []

    def clear_session(self, session_id: str, thread_id: str) -> None:
        metadata = self.get_session_meta(session_id)
        if not metadata:
            return

        metadata.update(
            {
                "thread_id": thread_id,
                "message_count": 0,
                "figure_count": 0,
            }
        )
        self.upsert_session_meta(session_id, metadata)
        self.client.delete(self._messages_key(session_id))
        self.client.delete(self._figures_key(session_id))

    def delete_session(self, session_id: str) -> None:
        self.client.delete(
            self._meta_key(session_id),
            self._messages_key(session_id),
            self._figures_key(session_id),
        )
        self.client.srem(self.active_sessions_key, session_id)
