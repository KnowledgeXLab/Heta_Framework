"""Freeze paired graph/text objects and stage exact native seed rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from heta_framework.common.stores.object.types import join_object_key
from heta_framework.generation.types import ObjectInput

from .types import GraphDocument, GraphInputError, PreparedGraphInputs


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


async def from_objects(request, store, config):
    pairs = {}
    reads = []
    for slot, suffix in (("kg", ".json"), ("text", ".txt")):
        selection = request.inputs[slot]
        assert isinstance(selection, ObjectInput)
        keys = selection.keys or tuple(item.key for item in await store.list(selection.prefix))
        keys = [key for key in keys if key.endswith(suffix)]
        if not keys:
            raise GraphInputError(f"empty {slot} selection")
        for key in sorted(keys):
            # Explicit basename pairing replaces fuzzy paper-title matching.
            identity = Path(key).stem
            entry = pairs.setdefault(identity, {})
            if slot in entry:
                raise GraphInputError(f"duplicate {slot} basename: {identity}")
            raw = await store.get(key)
            entry[slot] = json.loads(raw) if slot == "kg" else raw.decode("utf-8")
            reads.append({"key": key, "sha256": digest(raw), "input": slot})
    documents = []
    for identity, pair in pairs.items():
        if set(pair) != {"kg", "text"}:
            raise GraphInputError(f"missing graph/text partner: {identity}")
        graph, text = pair["kg"], pair["text"]
        validate_graph(graph, text)
        title = graph["A_Meta_Factual_Entities"]["Paper"]["title"]
        documents.append(GraphDocument(identity, title, text, graph, {"pair_id": identity}))
    if len(documents) > config.max_documents:
        raise GraphInputError("object selection exceeds max_documents; select explicit inputs")
    return documents, {"objects_read": reads}


def validate_graph(graph, text):
    """Check paired input integrity before paying for native scientific review."""
    from ._native.domain.graph_screen import B_NAME_FIELDS, _normalize_name

    if not isinstance(graph, dict) or not isinstance(text, str) or not text.strip():
        raise GraphInputError("expected a KG object and nonempty full text")
    title = graph.get("A_Meta_Factual_Entities", {}).get("Paper", {}).get("title")
    groups = graph.get("B_Textually_Mentioned_Entities")
    relations = graph.get("E_Relations")
    if not isinstance(title, str) or not title.strip() or not isinstance(groups, dict):
        raise GraphInputError("KG requires Paper.title and textually mentioned entity groups")
    if not isinstance(relations, list) or not relations:
        raise GraphInputError("KG requires nonempty E_Relations")
    names = set()
    lines = text.splitlines(keepends=True)

    def check_evidence(item):
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise GraphInputError("graph assertion requires evidence with source_lines and quote")
        for cite in evidence:
            span = cite.get("source_lines", [])
            if (
                len(span) != 2
                or any(type(n) is not int for n in span)
                or not 1 <= span[0] <= span[1] <= len(lines)
                or not cite.get("quote")
                or cite["quote"] != "".join(lines[span[0] - 1 : span[1]])
            ):
                raise GraphInputError("graph evidence does not match paired full text")

    for category, records in groups.items():
        if category not in B_NAME_FIELDS or not isinstance(records, list):
            raise GraphInputError(f"unsupported native entity group: {category}")
        for item in records:
            name = item.get(B_NAME_FIELDS[category])
            if not isinstance(name, str) or not name.strip():
                raise GraphInputError("entity lacks its native name field")
            names.add(_normalize_name(name))
            check_evidence(item)
    for item in relations:
        if any(_normalize_name(item.get(endpoint)) not in names for endpoint in ("head", "tail")):
            raise GraphInputError("relation endpoint is absent from textually mentioned entities")
        if not isinstance(item.get("relation"), str) or not item["relation"].strip():
            raise GraphInputError("relation type must be nonempty")
        check_evidence(item)


async def freeze(documents, provenance, request, store):
    # Validate the entire selection before writing any prepared document.
    if len({document.document_id for document in documents}) != len(documents):
        raise GraphInputError("duplicate document identity in input selection")
    for document in documents:
        validate_graph(document.graph, document.text)
    entries = []
    for document in documents:
        identity = digest(document.document_id.encode())[:24]
        files = {}
        for slot, suffix, raw in (
            ("kg", "json", encode(document.graph)),
            ("text", "txt", document.text.encode()),
        ):
            key = join_object_key(request.output_prefix, f"inputs/{slot}/{identity}.{suffix}")
            await store.put(key, raw)
            files[slot] = {"key": key, "sha256": digest(raw)}
        entries.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "provenance": document.provenance,
                **files,
            }
        )
    key = join_object_key(request.output_prefix, "inputs/manifest.json")
    await store.put(
        key,
        encode(
            {
                "schema_version": 1,
                "format": "omega_graph_inputs",
                "documents": entries,
                "provenance": provenance,
            }
        ),
    )
    return PreparedGraphInputs(key, len(entries))


async def stage_snapshot(prepared, store, root):
    manifest = json.loads(await store.get(prepared.manifest_key))
    rows = []
    for index, document in enumerate(manifest["documents"]):
        directory = root / str(index)
        directory.mkdir(parents=True)
        paths = {}
        for slot, filename in (("kg", "kg.json"), ("text", "paper.txt")):
            ref = document[slot]
            raw = await store.get(ref["key"])
            if digest(raw) != ref["sha256"]:
                raise GraphInputError(f"input snapshot changed: {ref['key']}")
            paths[slot] = directory / filename
            paths[slot].write_bytes(raw)
        rows.append(
            {
                "paper_id": digest(
                    (document["kg"]["sha256"] + document["text"]["sha256"]).encode()
                )[:24],
                "row_index": index,
                "title": document["title"],
                "kg_path": str(paths["kg"]),
                "kg_sha256": document["kg"]["sha256"],
                "paper_dir": str(directory),
                "paper_text_path": str(paths["text"]),
                "paper_text_chars": len(paths["text"].read_text()),
            }
        )
    return rows
