"""Resolve query provenance to existing KB artifacts, without rerunning extraction."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from functools import partial
from typing import TYPE_CHECKING

from heta_framework.kb.chunking.types import ParsedChunk
from heta_framework.kb.graphing.types import ExtractedEntity, ExtractedRelation
from heta_framework.kb.parsing.types import ParsedDocument
from heta_framework.kb.search import QueryRequest
from heta_framework.kb.steps import store_ref

from .types import GraphDocument, GraphInputError

if TYPE_CHECKING:
    from heta_framework.kb import KnowledgeBase

    from .config import HetaGenTaskProducerGraphConfig


# These are semantic correspondences, not a fallback classification of unknown entities.
ENTITY_FIELDS = {
    "TASK": ("Tasks", "name"),
    "METHOD": ("Methods", "name"),
    "DATASET": ("Datasets", "name"),
    "METRIC": ("Metrics", "full_name"),
    "BASELINE": ("Baselines", "name"),
    "IMPLEMENTATION": ("Implementation", "framework"),
    "THEOREM": ("Theorems", "statement"),
    "DOMAIN_TERM": ("Domain_Terms", "term"),
}


async def from_knowledge_base(
    kb: KnowledgeBase, query: str, config: HetaGenTaskProducerGraphConfig
):
    if kb.run_record.status != "succeeded":
        raise GraphInputError("knowledge base must have a succeeded build")
    run_id = kb.run_record.run_id
    query_request = QueryRequest(
        text=query,
        mode=config.query_mode,
        top_k=config.top_k,
        options={"generate_answer": False, "only_need_context": True},
    )
    response = await kb.query(query_request)
    provenance = {
        "knowledge_base": kb.name,
        "run_id": run_id,
        "query": query,
        "query_mode": config.query_mode,
        "request": asdict(query_request),
        "response": asdict(response),
    }
    if not response.results and not response.citations:
        if response.answer:
            raise GraphInputError("query returned an answer without source evidence")
        return [], provenance

    reads = []

    async def load(artifact, decoder, selected_keys=None):
        keys = kb.run_record.artifacts.get(artifact)
        if not isinstance(keys, (list, tuple)) or not keys:
            raise GraphInputError(f"missing or empty KB artifact: {artifact}")
        if selected_keys is not None:
            if any(key not in keys for key in selected_keys):
                raise GraphInputError(f"query references an undeclared {artifact} object")
            keys = selected_keys
        producers = [
            step for step in kb.recipe.expanded_steps() if artifact in step.capabilities.artifacts
        ]
        if len(producers) > 1:
            raise GraphInputError(f"ambiguous producing step for artifact: {artifact}")
        refs = (
            [
                ref
                for ref in producers[0].requirements.components
                if ref.namespace == "stores" and ref.kind == "objects"
            ]
            if producers
            else [store_ref("objects")]
        )
        if len(refs) != 1:
            raise GraphInputError(f"ambiguous object store for artifact: {artifact}")
        store = kb.recipe.get_component(refs[0])
        records = []
        for key in keys:
            raw = await store.get(key)
            records.append(decoder(raw))
            reads.append(
                {
                    "artifact": artifact,
                    "store": refs[0].key,
                    "key": key,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        return records

    parsed = await load(config.parsed_documents_artifact, ParsedDocument.from_json)
    by_id = {doc.document_id: doc for doc in parsed}
    if len(by_id) != len(parsed):
        raise GraphInputError("duplicate document IDs in parsed documents")
    by_key = {}
    for doc in parsed:
        by_key.setdefault(doc.source.key, []).append(doc.document_id)
    selected = []
    wiki_origins = {}
    for result in (*response.results, *response.citations):
        source = result.source
        ids = list(source.get("document_ids", ()))
        if source.get("document_id"):
            ids.insert(0, source["document_id"])
        if source.get("object_type") == "wiki":
            origin = source.get("origin") or {}
            source_hash = result.metadata.get("origin_source_content_sha256")
            if not origin.get("object_key"):
                from heta_framework.kb._wiki import parse_wiki_page

                page_key = source.get("object_key")
                if not isinstance(page_key, str) or not page_key:
                    raise GraphInputError("wiki result lacks its page object key")
                if page_key not in wiki_origins:
                    pages = await load(
                        "wiki_page_keys",
                        partial(parse_wiki_page, key=page_key),
                        selected_keys=(page_key,),
                    )
                    wiki_origins[page_key] = pages[0].source
                page_source = wiki_origins[page_key]
                origin = {"object_key": page_source.key}
                source_hash = page_source.content_sha256
            original_ids = by_key.get(origin["object_key"], [])
            if len(original_ids) != 1:
                raise GraphInputError("wiki origin cannot identify one original document")
            original = by_id[original_ids[0]]
            if source_hash and source_hash != original.source.content_sha256:
                raise GraphInputError("wiki origin content hash differs from parsed document")
            if any(doc_id in by_id and doc_id not in original_ids for doc_id in ids):
                raise GraphInputError("wiki origin conflicts with original document identity")
            ids = original_ids
        if not ids:
            keys = list(source.get("object_keys", ())) + list(source.get("source_keys", ()))
            if source.get("object_key"):
                keys.insert(0, source["object_key"])
            for key in keys:
                matches = by_key.get(key, [])
                if len(matches) != 1:
                    raise GraphInputError(f"query source cannot identify one document: {key}")
                ids.extend(matches)
        if not ids:
            raise GraphInputError("query result lacks resolvable document provenance")
        for doc_id in ids:
            if doc_id not in by_id:
                raise GraphInputError(f"query references unknown document: {doc_id}")
            if doc_id not in selected:
                selected.append(doc_id)
    entities = await load(config.entities_artifact, ExtractedEntity.from_json)
    relations = await load(config.relations_artifact, ExtractedRelation.from_json)
    chunks = await load(config.chunks_artifact, ParsedChunk.from_json)
    documents = []
    for doc_id in selected[: config.max_documents]:
        doc = by_id[doc_id]
        doc_entities = [item for item in entities if item.document_id == doc_id]
        doc_relations = [item for item in relations if item.document_id == doc_id]
        doc_chunks = [item for item in chunks if item.document_id == doc_id]
        if not doc_entities or not doc_relations or not doc_chunks:
            raise GraphInputError(f"document lacks paired graph/text artifacts: {doc_id}")
        text = (
            doc.original_content.text
            if doc.original_content
            else "\n\n".join(
                page.text for page in sorted(doc.pages, key=lambda page: page.page_index)
            )
        )
        if any(chunk.text not in text for chunk in doc_chunks):
            text = "\n\n".join(
                page.text for page in sorted(doc.pages, key=lambda page: page.page_index)
            )
        graph = make_graph(
            doc, text, doc_entities, doc_relations, doc_chunks, config.entity_type_mapping
        )
        documents.append(
            GraphDocument(
                doc_id,
                doc.source.name,
                text,
                graph,
                {"document_id": doc_id, "source": asdict(doc.source)},
            )
        )
    if kb.run_record.run_id != run_id:
        raise GraphInputError("knowledge base build changed during input preparation")
    provenance["objects_read"] = reads
    provenance["entity_type_mapping"] = dict(config.entity_type_mapping)
    return documents, provenance


def make_graph(doc, text, entities, relations, chunks, entity_type_mapping=None):
    """Preserve entity types and cite literal source lines for every graph assertion."""
    from ._native.domain.graph_screen import B_NAME_FIELDS, _normalize_name

    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunk_map) != len(chunks):
        raise GraphInputError("duplicate chunk IDs")
    lines = text.splitlines(keepends=True)

    def evidence(item):
        output = []
        for chunk_id in item.source_chunk_ids:
            chunk = chunk_map.get(chunk_id)
            if chunk is None or not chunk.text.strip():
                raise GraphInputError(f"missing source chunk: {chunk_id}")
            start = text.find(chunk.text)
            if start < 0 or text.find(chunk.text, start + 1) >= 0:
                raise GraphInputError(f"chunk is absent or ambiguous in full text: {chunk_id}")
            first = text[:start].count("\n")
            last = text[: start + len(chunk.text) - 1].count("\n")
            output.append(
                {
                    "source_lines": [first + 1, last + 1],
                    "quote": "".join(lines[first : last + 1]),
                    "chunk_id": chunk_id,
                    "document_id": doc.document_id,
                }
            )
        if not output:
            raise GraphInputError("graph assertion lacks source chunks")
        return output

    groups, entity_map, names = {}, {}, {}
    for entity in entities:
        semantic_type = entity.type.upper()
        mapping = entity_type_mapping or {}
        if semantic_type not in ENTITY_FIELDS and semantic_type not in mapping:
            raise GraphInputError(f"unsupported entity type: {entity.type}")
        normalized = _normalize_name(entity.name)
        # Preserve separate mention descriptions/evidence under the same typed name.
        signature = (entity_type_mapping or {}).get(semantic_type, semantic_type)
        if normalized in names and names[normalized] != signature:
            raise GraphInputError(f"ambiguous entity name: {entity.name}")
        names[normalized] = signature
        if entity.entity_id in entity_map:
            raise GraphInputError(f"duplicate entity ID: {entity.entity_id}")
        entity_map[entity.entity_id] = entity
        if semantic_type in mapping:
            group = mapping[semantic_type]
            field = B_NAME_FIELDS[group]
        else:
            group, field = ENTITY_FIELDS[semantic_type]
        groups.setdefault(group, []).append(
            {
                field: entity.name,
                "entity_id": entity.entity_id,
                "description": entity.description,
                "attributes": dict(entity.attributes),
                "evidence": evidence(entity),
            }
        )
    edges = []
    for relation in relations:
        head = entity_map.get(relation.source_entity_id)
        tail = entity_map.get(relation.target_entity_id)
        if head is None or tail is None:
            raise GraphInputError(
                f"relation has missing or cross-document endpoint: {relation.relation_id}"
            )
        if head.name != relation.source_entity_name or tail.name != relation.target_entity_name:
            raise GraphInputError(f"relation endpoint name mismatch: {relation.relation_id}")
        edges.append(
            {
                "head": head.name,
                "tail": tail.name,
                "relation": relation.type,
                "head_type": head.type,
                "tail_type": tail.type,
                "description": relation.description,
                "evidence": evidence(relation),
                "relation_id": relation.relation_id,
            }
        )
    return {
        "A_Meta_Factual_Entities": {"Paper": {"title": doc.source.name}},
        "B_Textually_Mentioned_Entities": groups,
        "E_Relations": edges,
    }
