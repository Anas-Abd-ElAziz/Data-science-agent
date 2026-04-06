"""Dataset storage backends for uploaded tabular files."""

from __future__ import annotations

from datetime import datetime, timezone
import mimetypes
import os
from pathlib import Path
import re
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def _safe_filename(filename: str) -> str:
    candidate = Path(filename).name or "dataset"
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-")
    return candidate or "dataset"


class DatasetStoreError(RuntimeError):
    """Base dataset storage error."""


class DatasetStoreConfigError(DatasetStoreError):
    """Raised when dataset storage is not configured."""


class DatasetNotFoundError(DatasetStoreError):
    """Raised when a stored dataset is no longer available."""


class S3DatasetStore:
    """S3-backed storage for uploaded dataset bytes."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "datasets",
        client=None,
        delete_on_session_delete: bool = False,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/") or "datasets"
        self.client = client or boto3.client("s3")
        self.delete_on_session_delete = delete_on_session_delete

    @classmethod
    def from_env(cls) -> "S3DatasetStore | None":
        bucket = os.getenv("S3_DATASET_BUCKET")
        if not bucket:
            return None

        region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        client = boto3.client("s3", region_name=region_name)
        return cls(
            bucket=bucket,
            prefix=os.getenv("S3_DATASET_PREFIX", "datasets"),
            client=client,
            delete_on_session_delete=(
                (os.getenv("S3_DELETE_ON_SESSION_DELETE") or "").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
        )

    def _build_key(self, session_id: str, filename: str) -> str:
        safe_filename = _safe_filename(filename)
        return f"{self.prefix}/{session_id}/{safe_filename}"

    def ping(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except (BotoCoreError, ClientError):
            return False
        return True

    def put_dataset(
        self, *, session_id: str, file_bytes: bytes, filename: str, sha256: str
    ) -> dict[str, Any]:
        key = self._build_key(session_id=session_id, filename=filename)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        try:
            response = self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=file_bytes,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError) as exc:
            raise DatasetStoreError("Failed to upload dataset to S3.") from exc

        return {
            "storage_backend": "s3",
            "bucket": self.bucket,
            "key": key,
            "filename": filename,
            "size": len(file_bytes),
            "sha256": sha256,
            "etag": str(response.get("ETag", "")).strip('"') or None,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_dataset_bytes(self, dataset_ref: dict[str, Any]) -> bytes:
        bucket = dataset_ref.get("bucket") or self.bucket
        key = dataset_ref.get("key")
        if not key:
            raise DatasetStoreConfigError(
                "Dataset reference is missing the S3 object key."
            )

        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in {"404", "NoSuchBucket", "NoSuchKey", "NotFound"}:
                raise DatasetNotFoundError(
                    "Dataset is no longer available in S3."
                ) from exc
            raise DatasetStoreError("Failed to download dataset from S3.") from exc
        except BotoCoreError as exc:
            raise DatasetStoreError("Failed to download dataset from S3.") from exc

        return response["Body"].read()

    def delete_dataset(self, dataset_ref: dict[str, Any]) -> None:
        if not self.delete_on_session_delete:
            return

        bucket = dataset_ref.get("bucket") or self.bucket
        key = dataset_ref.get("key")
        if not key:
            return

        try:
            self.client.delete_object(Bucket=bucket, Key=key)
        except (BotoCoreError, ClientError):
            return
