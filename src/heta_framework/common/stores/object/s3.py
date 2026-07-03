"""S3-compatible object store implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from heta_framework.common.stores.object.types import (
    ObjectInfo,
    S3AddressingStyle,
    join_object_key,
    strip_object_prefix,
    validate_object_key,
    validate_object_prefix,
)


@dataclass(frozen=True)
class S3ObjectStoreConfig:
    """Configuration for S3-compatible object stores."""

    bucket: str
    prefix: str = ""
    endpoint_url: str | None = None
    region: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    addressing_style: S3AddressingStyle = "auto"
    signature_version: str | None = None
    verify: bool | str = True

    def __post_init__(self) -> None:
        if self.bucket.strip() == "":
            raise ValueError("bucket must not be empty")
        validate_object_prefix(self.prefix)


class S3ObjectStore:
    """Object store backed by an S3-compatible service."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        addressing_style: S3AddressingStyle = "auto",
        signature_version: str | None = None,
        verify: bool | str = True,
        client: Any | None = None,
    ) -> None:
        self.config = S3ObjectStoreConfig(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=endpoint_url,
            region=region,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            addressing_style=addressing_style,
            signature_version=signature_version,
            verify=verify,
        )
        self._client = client if client is not None else _create_s3_client(self.config)

    async def put(self, key: str, data: bytes) -> None:
        """Store bytes at a key."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        self._client.put_object(
            Bucket=self.config.bucket,
            Key=self._backend_key(key),
            Body=data,
        )

    async def get(self, key: str) -> bytes:
        """Read bytes from a key."""
        try:
            response = self._client.get_object(
                Bucket=self.config.bucket,
                Key=self._backend_key(key),
            )
        except Exception as exc:
            if _is_not_found_error(exc):
                raise FileNotFoundError(key) from exc
            raise
        return response["Body"].read()

    async def exists(self, key: str) -> bool:
        """Return whether a key exists."""
        try:
            self._client.head_object(
                Bucket=self.config.bucket,
                Key=self._backend_key(key),
            )
            return True
        except Exception as exc:
            if _is_not_found_error(exc):
                return False
            raise

    async def list(self, prefix: str = "") -> list[ObjectInfo]:
        """List objects under a prefix."""
        backend_prefix = self._backend_prefix(prefix)
        paginator = self._client.get_paginator("list_objects_v2")

        objects: list[ObjectInfo] = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=backend_prefix):
            for item in page.get("Contents", []):
                key = strip_object_prefix(self.config.prefix, item["Key"])
                if key == "":
                    continue
                objects.append(
                    ObjectInfo(
                        key=key,
                        size=item.get("Size"),
                        modified_at=item.get("LastModified"),
                        etag=_strip_etag_quotes(item.get("ETag")),
                    )
                )
        objects.sort(key=lambda item: item.key)
        return objects

    async def delete(self, key: str) -> None:
        """Delete a key if it exists."""
        self._client.delete_object(
            Bucket=self.config.bucket,
            Key=self._backend_key(key),
        )

    async def aclose(self) -> None:
        """Release resources held by the store."""
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def _backend_key(self, key: str) -> str:
        return join_object_key(self.config.prefix, validate_object_key(key))

    def _backend_prefix(self, prefix: str) -> str:
        normalized_prefix = validate_object_prefix(prefix)
        if normalized_prefix == "":
            normalized_store_prefix = validate_object_prefix(self.config.prefix)
            return f"{normalized_store_prefix.rstrip('/')}/" if normalized_store_prefix else ""
        return f"{join_object_key(self.config.prefix, normalized_prefix).rstrip('/')}/"


def _create_s3_client(config: S3ObjectStoreConfig) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError("boto3 is not installed; install the `heta-framework[s3]` extra") from exc

    client_config: dict[str, Any] = {
        "s3": {"addressing_style": config.addressing_style},
    }
    if config.signature_version is not None:
        client_config["signature_version"] = config.signature_version

    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        region_name=config.region,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        aws_session_token=config.session_token,
        verify=config.verify,
        config=Config(**client_config),
    )


def _is_not_found_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error", {})
    code = str(error.get("Code", ""))
    return code in {"404", "NoSuchKey", "NotFound"}


def _strip_etag_quotes(etag: str | None) -> str | None:
    if etag is None:
        return None
    return etag.strip('"')
