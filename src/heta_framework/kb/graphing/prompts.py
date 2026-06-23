"""Prompts for graph-building steps."""

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge graph entity extraction engine.
Return only valid JSON. Do not include markdown, explanations, or extra text."""

ENTITY_EXTRACTION_PROMPT = """Extract knowledge graph entities from the chunk below.

Rules:
- Return a JSON object with exactly one top-level key: "entities".
- "entities" must be an array.
- Each entity must include: name, type, subtype, description, attributes.
- name must be a specific named entity, not a pronoun or vague phrase.
- type must be a concise entity category.
- subtype may be null when no reliable subtype is available.
- description must be a short factual description grounded in the chunk.
- attributes must be a JSON object of string keys and string values.
- Do not invent facts that are not supported by the chunk.
- If no reliable entities exist, return {{"entities": []}}.

Chunk metadata:
- chunk_id: {chunk_id}
- document_id: {document_id}
- source: {source_name}

Chunk text:
{chunk_text}
"""

ENTITY_EXTRACTION_RETRY_PROMPT = """The previous entity extraction response was invalid.

Validation error:
{error}

Return the corrected result for the same chunk. Return only valid JSON with this shape:
{{"entities":[{{"name":"...","type":"...","subtype":null,"description":"...","attributes":{{}}}}]}}

Chunk metadata:
- chunk_id: {chunk_id}
- document_id: {document_id}
- source: {source_name}

Chunk text:
{chunk_text}
"""

RELATION_EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge graph relation extraction engine.
Return only valid JSON. Do not include markdown, explanations, or extra text."""

RELATION_EXTRACTION_PROMPT = """Extract knowledge graph relations from the chunk below.

Rules:
- Return a JSON object with exactly one top-level key: "relations".
- "relations" must be an array.
- Each relation must include: source, target, type, name, description, attributes.
- source and target must exactly match names from the provided entities.
- Do not create new entities.
- Do not create self-relations.
- type must be a concise relation category.
- name must be the specific relation name.
- description must be a short factual description grounded in the chunk.
- attributes must be a JSON object of string keys and string values.
- Do not invent facts that are not supported by the chunk.
- If no reliable relations exist, return {{"relations": []}}.

Chunk metadata:
- chunk_id: {chunk_id}
- document_id: {document_id}
- source: {source_name}

Entities:
{entities_json}

Chunk text:
{chunk_text}
"""

RELATION_EXTRACTION_RETRY_PROMPT = """The previous relation extraction response was invalid.

Validation error:
{error}

Return the corrected result for the same chunk. Return only valid JSON with this shape:
{{"relations":[{{"source":"...","target":"...","type":"...","name":"...","description":"...","attributes":{{}}}}]}}

Remember:
- source and target must exactly match names from the provided entities.
- Do not create new entities.
- Do not create self-relations.

Chunk metadata:
- chunk_id: {chunk_id}
- document_id: {document_id}
- source: {source_name}

Entities:
{entities_json}

Chunk text:
{chunk_text}
"""

ENTITY_DEDUPLICATION_SYSTEM_PROMPT = """You are a precise knowledge graph entity deduplication engine.
Return only valid JSON. Do not include markdown, explanations, or extra text."""

ENTITY_DEDUPLICATION_PROMPT = """Merge duplicate knowledge graph entities.

Rules:
- Return a JSON object with exactly one top-level key: "entity".
- The entity must include: name, type, subtype, description, attributes.
- Preserve only facts supported by the input entities.
- Prefer the clearest canonical name.
- description must be concise but include the useful facts from all duplicates.
- attributes must be a JSON object of string keys and string values.

Entities:
{entities_json}
"""

ENTITY_DEDUPLICATION_RETRY_PROMPT = """The previous entity deduplication response was invalid.

Validation error:
{error}

Return the corrected result for the same entities. Return only valid JSON with this shape:
{{"entity":{{"name":"...","type":"...","subtype":null,"description":"...","attributes":{{}}}}}}

Entities:
{entities_json}
"""

RELATION_DEDUPLICATION_SYSTEM_PROMPT = """You are a precise knowledge graph relation deduplication engine.
Return only valid JSON. Do not include markdown, explanations, or extra text."""

RELATION_DEDUPLICATION_PROMPT = """Merge duplicate knowledge graph relations.

Rules:
- Return a JSON object with exactly one top-level key: "relation".
- The relation must include: type, name, description, attributes.
- Preserve only facts supported by the input relations.
- description must be concise but include the useful facts from all duplicates.
- attributes must be a JSON object of string keys and string values.
- Do not change the relation endpoints.

Relations:
{relations_json}
"""

RELATION_DEDUPLICATION_RETRY_PROMPT = """The previous relation deduplication response was invalid.

Validation error:
{error}

Return the corrected result for the same relations. Return only valid JSON with this shape:
{{"relation":{{"type":"...","name":"...","description":"...","attributes":{{}}}}}}

Relations:
{relations_json}
"""
