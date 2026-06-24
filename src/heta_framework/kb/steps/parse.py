"""Parse raw objects into ParsedDocument artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.parsing import DocumentParserRegistry, make_document_id, make_parsed_source
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, parser_ref, store_ref


@dataclass(frozen=True)
class ParseDocumentsConfig:
    """Configuration for ParseDocuments."""

    raw_prefix: str = "raw"
    parsed_prefix: str = "parsed"
    skip_unsupported: bool = True
    object_store: str | None = None
    parser_registry: str | None = None

    def __post_init__(self) -> None:
        validate_object_prefix(self.raw_prefix)
        validate_object_prefix(self.parsed_prefix)


@dataclass(frozen=True)
class ParseDocumentsResult:
    """Artifacts produced by ParseDocuments."""

    document_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...] = ()


class ParseDocuments:
    """Parse raw files from an object store into ParsedDocument JSON objects."""

    name = "parse_documents"

    def __init__(self, config: ParseDocumentsConfig | None = None) -> None:
        self.config = config or ParseDocumentsConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    parser_ref(self.config.parser_registry),
                }
            )
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset({"parse_documents_result", "parsed_document_keys"})
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return parsed document objects produced by this step."""
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                "parsed_document_keys",
                component=store_ref("objects", self.config.object_store).key,
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run the parse step and store parsed documents as JSON bytes."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        parser_registry = _require_parser_registry(
            context.get_component(parser_ref(self.config.parser_registry).key)
        )

        document_keys: list[str] = []
        skipped_keys: list[str] = []
        for item in await object_store.list(self.config.raw_prefix):
            file_type = _file_type_from_key(item.key)
            if file_type is None or parser_registry.find_parser(file_type) is None:
                if self.config.skip_unsupported:
                    skipped_keys.append(item.key)
                    continue
                raise ValueError(f"no parser registered for raw object: {item.key}")

            data = await object_store.get(item.key)
            source = make_parsed_source(
                key=item.key,
                name=PurePosixPath(item.key).name,
                file_type=file_type,
                data=data,
            )
            document_key = join_object_key(
                self.config.parsed_prefix,
                f"{make_document_id(source.content_sha256)}.json",
            )
            if await object_store.exists(document_key):
                document_keys.append(document_key)
                continue

            document = await parser_registry.parse(source, data)
            await object_store.put(document_key, document.to_json_bytes())
            document_keys.append(document_key)

        result = ParseDocumentsResult(
            document_keys=tuple(document_keys),
            skipped_keys=tuple(skipped_keys),
        )
        context.set_artifact("parse_documents_result", result)
        context.set_artifact("parsed_document_keys", result.document_keys)


def _file_type_from_key(key: str) -> str | None:
    suffix = PurePosixPath(key).suffix.lower().lstrip(".")
    return suffix or None


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_parser_registry(component: object) -> DocumentParserRegistry:
    if not isinstance(component, DocumentParserRegistry):
        raise TypeError("parsers.documents must be a DocumentParserRegistry")
    return component
