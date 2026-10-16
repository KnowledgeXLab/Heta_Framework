"""Separate non-secret run policy from API credentials."""

import os
import re
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: str = "mock"
    tasks_per_package: int = Field(default=1, ge=1, le=100)
    max_revisions: int = Field(default=1, ge=0, le=10)
    max_role_attempts: int = Field(default=2, ge=1, le=3)
    max_concurrency: int = Field(default=1, ge=1)
    role_timeout_sec: int = Field(default=1800, ge=30)
    validation_timeout_sec: int = Field(default=1800, ge=30)
    codex_version: str = Field(default="0.154.0", pattern=r"^\d+\.\d+\.\d+$")
    role_image: str = Field(
        default="sci-pipeline/role:codex-0.154.0", pattern=r"^[A-Za-z0-9][A-Za-z0-9._/@:-]*$"
    )
    docker_builder: str = Field(default="default", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    environment_cpus: int = Field(default=2, ge=1)
    environment_memory_mb: int = Field(default=4096, ge=512)
    job_overhead_timeout_sec: int = Field(default=1200, ge=60)
    worker_stop_grace_sec: int = Field(default=30, ge=1, le=120)
    mock_revise_once: bool = False
    generate_skills: bool = False
    generation_guidance: str = ""
    difficulty_guidance: str = ""

    @field_validator("mode")
    @classmethod
    def known_mode(cls, value: str) -> str:
        if value not in {"mock", "harbor"}:
            raise ValueError("mode must be mock or harbor")
        return value


def read_environment(env_file: Path) -> dict[str, str | None]:
    """Read one explicit dotenv file; exported variables take precedence."""
    return {**dotenv_values(env_file, interpolate=False), **os.environ}


def validate_endpoint(address: str | None) -> str | None:
    if address:
        from urllib.parse import urlsplit

        parsed = urlsplit(address)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.query
        ):
            raise ValueError("Endpoint must use HTTP(S), without embedded credentials or query")
    return address


class NetworkSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    container_proxy: str | None = None
    container_no_proxy: str = "localhost,127.0.0.1"
    docker_network_pool: str | None = None

    @field_validator("docker_network_pool")
    @classmethod
    def private_network_pool(cls, value: str | None) -> str | None:
        if value:
            from ipaddress import IPv4Network

            pool = IPv4Network(value)
            private = any(
                pool.subnet_of(IPv4Network(cidr))
                for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
            )
            if not private or not 16 <= pool.prefixlen <= 28:
                raise ValueError("Docker network pool must be a private IPv4 /16 through /28")
            return str(pool)
        return None

    _validate_proxy = field_validator("container_proxy")(validate_endpoint)

    @classmethod
    def from_values(cls, values: dict) -> "NetworkSettings":
        return cls(
            docker_network_pool=(values.get("SCI_PIPELINE_DOCKER_NETWORK_POOL") or "").strip()
            or None,
            container_proxy=(values.get("SCI_PIPELINE_CONTAINER_PROXY") or "").strip() or None,
            container_no_proxy=values.get("SCI_PIPELINE_CONTAINER_NO_PROXY")
            or "localhost,127.0.0.1",
        )

    @classmethod
    def load(cls, env_file: Path) -> "NetworkSettings":
        return cls.from_values(read_environment(env_file))

    def container_environment(self) -> dict[str, str]:
        if not self.container_proxy:
            return {}
        return {
            "HTTP_PROXY": self.container_proxy,
            "HTTPS_PROXY": self.container_proxy,
            "http_proxy": self.container_proxy,
            "https_proxy": self.container_proxy,
            "NO_PROXY": self.container_no_proxy,
            "no_proxy": self.container_no_proxy,
        }


class ApiSettings(NetworkSettings):
    api_key: SecretStr
    model: str
    base_url: str | None = None

    _validate_base = field_validator("base_url")(validate_endpoint)

    @field_validator("model")
    @classmethod
    def model_identifier(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", value):
            raise ValueError("Model must be a plain API model identifier")
        return value

    @classmethod
    def load(cls, env_file: Path) -> "ApiSettings":
        values = read_environment(env_file)
        required = ("OPENAI_API_KEY", "SCI_PIPELINE_MODEL")
        missing = [name for name in required if not (values.get(name) or "").strip()]
        if missing:
            raise ValueError("Missing settings: " + ", ".join(missing))
        return cls(
            api_key=values["OPENAI_API_KEY"],
            model=values["SCI_PIPELINE_MODEL"].strip(),
            base_url=(values.get("OPENAI_BASE_URL") or "").strip() or None,
            **NetworkSettings.from_values(values).model_dump(),
        )

    def environment(self) -> dict[str, str]:
        result = {"OPENAI_API_KEY": self.api_key.get_secret_value()}
        if self.base_url:
            result["OPENAI_BASE_URL"] = self.base_url
        return result
