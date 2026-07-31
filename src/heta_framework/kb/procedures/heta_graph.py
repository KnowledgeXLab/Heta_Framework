"""Heta-style graph build procedure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from heta_framework.kb.steps import (
    BuildGraph,
    BuildGraphConfig,
    ConstrainGraphByOntology,
    ConstrainGraphByOntologyConfig,
    DeduplicateEntities,
    DeduplicateEntitiesConfig,
    DeduplicateRelations,
    DeduplicateRelationsConfig,
    ExtractUniversalGraph,
    ExtractUniversalGraphConfig,
    GraphTableNames,
    GraphVectorCollections,
    KnowledgeStepProtocol,
    MergeGraphIntoStore,
    MergeGraphIntoStoreConfig,
)

GraphProcedureMode = Literal["build", "merge_into_store"]


@dataclass(frozen=True)
class HetaGraphProcedure:
    """Static step composition for Heta-style graph construction."""

    mode: GraphProcedureMode = "build"
    deduplicate: bool = True
    schema_constraint_enabled: bool = False
    ontology_schema: Mapping[str, Any] = field(default_factory=dict)

    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "entity_keys"
    relation_keys_artifact: str = "relation_keys"
    deduplicated_entity_keys_artifact: str = "deduplicated_entity_keys"
    deduplicated_relation_keys_artifact: str = "deduplicated_relation_keys"

    table_names: GraphTableNames = field(default_factory=GraphTableNames)
    vector_collections: GraphVectorCollections = field(default_factory=GraphVectorCollections)

    object_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    language_model: str | None = None
    embedding_model: str | None = None

    @property
    def name(self) -> str:
        """Return the stable procedure name."""
        return "heta_graph"

    @classmethod
    def build(cls, **kwargs: Any) -> "HetaGraphProcedure":
        """Create a procedure that writes the current graph facts directly."""
        return cls(mode="build", **kwargs)

    @classmethod
    def merge_into_store(cls, **kwargs: Any) -> "HetaGraphProcedure":
        """Create a procedure that incrementally merges graph facts into storage."""
        return cls(mode="merge_into_store", **kwargs)

    def steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        """Expand this procedure into executable build steps."""
        constrained_entity_keys_artifact = "ontology_entity_keys"
        constrained_relation_keys_artifact = "ontology_relation_keys"
        base_entity_input = (
            constrained_entity_keys_artifact
            if self.schema_constraint_enabled
            else self.entity_keys_artifact
        )
        base_relation_input = (
            constrained_relation_keys_artifact
            if self.schema_constraint_enabled
            else self.relation_keys_artifact
        )
        entity_input = (
            self.deduplicated_entity_keys_artifact if self.deduplicate else base_entity_input
        )
        relation_input = (
            self.deduplicated_relation_keys_artifact if self.deduplicate else base_relation_input
        )

        steps: list[KnowledgeStepProtocol] = [
            ExtractUniversalGraph(
                ExtractUniversalGraphConfig(
                    chunk_keys_artifact=self.chunk_keys_artifact,
                    entity_keys_artifact=self.entity_keys_artifact,
                    relation_keys_artifact=self.relation_keys_artifact,
                    object_store=self.object_store,
                    language_model=self.language_model,
                )
            )
        ]
        if self.schema_constraint_enabled:
            steps.append(
                ConstrainGraphByOntology(
                    ConstrainGraphByOntologyConfig(
                        schema=self.ontology_schema,
                        chunk_keys_artifact=self.chunk_keys_artifact,
                        entity_keys_artifact=self.entity_keys_artifact,
                        relation_keys_artifact=self.relation_keys_artifact,
                        constrained_entity_keys_artifact=constrained_entity_keys_artifact,
                        constrained_relation_keys_artifact=constrained_relation_keys_artifact,
                        object_store=self.object_store,
                        language_model=self.language_model,
                    )
                )
            )

        if self.deduplicate:
            steps.extend(
                [
                    DeduplicateEntities(
                        DeduplicateEntitiesConfig(
                            entity_keys_artifact=base_entity_input,
                            deduplicated_entity_keys_artifact=(
                                self.deduplicated_entity_keys_artifact
                            ),
                            object_store=self.object_store,
                            language_model=self.language_model,
                            embedding_model=self.embedding_model,
                        )
                    ),
                    DeduplicateRelations(
                        DeduplicateRelationsConfig(
                            relation_keys_artifact=base_relation_input,
                            deduplicated_relation_keys_artifact=(
                                self.deduplicated_relation_keys_artifact
                            ),
                            object_store=self.object_store,
                            language_model=self.language_model,
                            embedding_model=self.embedding_model,
                        )
                    ),
                ]
            )

        if self.mode == "build":
            steps.append(
                BuildGraph(
                    BuildGraphConfig(
                        entity_keys_artifact=entity_input,
                        relation_keys_artifact=relation_input,
                        chunk_keys_artifact=self.chunk_keys_artifact,
                        table_names=self.table_names,
                        vector_collections=self.vector_collections,
                        object_store=self.object_store,
                        sql_store=self.sql_store,
                        vector_store=self.vector_store,
                        embedding_model=self.embedding_model,
                    )
                )
            )
        elif self.mode == "merge_into_store":
            steps.append(
                MergeGraphIntoStore(
                    MergeGraphIntoStoreConfig(
                        entity_keys_artifact=entity_input,
                        relation_keys_artifact=relation_input,
                        chunk_keys_artifact=self.chunk_keys_artifact,
                        table_names=self.table_names,
                        vector_collections=self.vector_collections,
                        object_store=self.object_store,
                        sql_store=self.sql_store,
                        vector_store=self.vector_store,
                        embedding_model=self.embedding_model,
                        language_model=self.language_model,
                    )
                )
            )
        else:
            raise ValueError(f"unsupported graph procedure mode: {self.mode}")

        return tuple(steps)
