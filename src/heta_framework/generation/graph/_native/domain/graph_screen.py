# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural screen: build the E-relation graph and assess its largest component.

This is a re-implementation (not a fork) of Omega's ``graph_screen``. It is a
pure function of the KG object plus thresholds: no network, no model, no IO
except :func:`assess_file`. Module A is read only for paper identity, module D is
intentionally ignored, and B/C labels annotate E endpoints without ever creating
isolated nodes — preserving Omega's gate semantics exactly.

The returned dict shapes intentionally match Omega's artefacts key-for-key so the
two implementations can be diffed in tests.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from heta_framework.generation.graph._native.config import ScreenThresholds
from heta_framework.generation.graph._native.io_utils import canonical_json_sha256, sha256_file

MISSING = {"", "n/a", "na", "none", "null", "unknown"}
MODULE_A = "A_Meta_Factual_Entities"
MODULE_B = "B_Textually_Mentioned_Entities"
MODULE_C = "C_Implicit_Abstracted_Entities"
MODULE_D = "D_Reference"
MODULE_E = "E_Relations"

# Which record field carries the human-readable label for each B category.
B_NAME_FIELDS: dict[str, str] = {
    "Tasks": "name",
    "Methods": "name",
    "Datasets": "name",
    "Metrics": "full_name",
    "Baselines": "name",
    "Implementation": "framework",
    "Theorems": "statement",
    "Figures_Tables": "caption",
    "Domain_Terms": "term",
}

GRAPH_SCHEMA_VERSION = "omega.science.paper_kg_graph.v1"
ASSESSMENT_SCHEMA_VERSION = "omega.science.paper_kg_structure_assessment.v1"


class ScreenResult(BaseModel):
    assessment: dict[str, Any]
    graph: dict[str, Any]
    largest_component_graph: dict[str, Any]


def assess_file(path: str | Path, *, thresholds: ScreenThresholds) -> ScreenResult:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Scholar-KG paper must be a JSON object: {source}")
    return assess(value, thresholds=thresholds, source_path=str(source), source_sha256=sha256_file(source))


def assess(
    paper_kg: dict[str, Any],
    *,
    thresholds: ScreenThresholds,
    source_path: str = "",
    source_sha256: str = "",
) -> ScreenResult:
    paper = _paper_identity(paper_kg)
    source_text = str(source_path)
    source_id = canonical_json_sha256({"path": source_text, "sha256": source_sha256})[:24]
    aliases = _aliases(paper_kg.get("entity_aliases"))
    b_labels = _b_labels(paper_kg.get(MODULE_B), aliases)
    c_labels = _c_labels(paper_kg.get(MODULE_C), aliases)

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()
    raw_relations = paper_kg.get(MODULE_E) or []
    if not isinstance(raw_relations, list):
        raise ValueError(f"{MODULE_E} must be a list")
    for index, relation in enumerate(raw_relations):
        if not isinstance(relation, dict):
            continue
        raw_head = _text(relation.get("head"))
        raw_tail = _text(relation.get("tail"))
        raw_relation = _text(relation.get("relation"))
        if not raw_head or not raw_tail or not raw_relation:
            continue
        head = _canonical(raw_head, aliases)
        tail = _canonical(raw_tail, aliases)
        relation_name = _normalize_relation(raw_relation)
        edge_key = (head, relation_name, tail)
        if edge_key in edge_keys:
            continue
        edge_keys.add(edge_key)
        head_id = _node_id(head)
        tail_id = _node_id(tail)
        _upsert_node(nodes, head_id, head, raw_head, relation.get("head_type"), b_labels, c_labels)
        _upsert_node(nodes, tail_id, tail, raw_tail, relation.get("tail_type"), b_labels, c_labels)
        edges.append(
            {
                "id": f"edge_{hashlib.sha256('|'.join(edge_key).encode('utf-8')).hexdigest()[:20]}",
                "source": head_id,
                "target": tail_id,
                "head": nodes[head_id]["name"],
                "tail": nodes[tail_id]["name"],
                "relation": relation_name,
                "properties": {
                    key: item
                    for key, item in relation.items()
                    if key not in {"head", "tail", "relation", "head_type", "tail_type"}
                },
                "sourceid": source_id,
                "sourcefilepath": source_text,
                "source_relation_index": index,
            }
        )

    components = _components(nodes, edges)
    node_to_component: dict[str, str] = {}
    edge_to_component: dict[str, str] = {}
    component_rows = []
    for rank, component in enumerate(components, 1):
        component_id = f"component_{rank:04d}"
        for node_id in component["node_ids"]:
            node_to_component[node_id] = component_id
        for edge_id in component["edge_ids"]:
            edge_to_component[edge_id] = component_id
        component_rows.append({"component_id": component_id, "rank": rank, **component})
    for node_id, node in nodes.items():
        node["component_id"] = node_to_component.get(node_id, "")
        node["sourceid"] = source_id
        node["sourcefilepath"] = source_text
    for edge in edges:
        edge["component_id"] = edge_to_component.get(edge["id"], "")

    largest = component_rows[0] if component_rows else _empty_component()
    total_nodes = len(nodes)
    largest_node_ids = set(largest.get("node_ids") or [])
    largest_edges = [
        edge for edge in edges if edge["source"] in largest_node_ids and edge["target"] in largest_node_ids
    ]
    relation_types = sorted({edge["relation"] for edge in largest_edges})
    b_anchor_nodes = sorted(
        node["name"] for node_id, node in nodes.items() if node_id in largest_node_ids and node["b_categories"]
    )
    c_anchor_nodes = sorted(
        node["name"] for node_id, node in nodes.items() if node_id in largest_node_ids and node["c_categories"]
    )
    degree: dict[str, int] = defaultdict(int)
    for edge in largest_edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    hubs = sorted(
        ({"node_id": node_id, "name": nodes[node_id]["name"], "degree": value} for node_id, value in degree.items()),
        key=lambda row: (-row["degree"], row["name"].casefold()),
    )[:10]
    metrics = {
        "component_nodes": len(largest_node_ids),
        "component_edges": len(largest_edges),
        "component_node_fraction": len(largest_node_ids) / total_nodes if total_nodes else 0.0,
        "relation_types": len(relation_types),
        "b_anchor_nodes": len(b_anchor_nodes),
        "c_anchor_nodes": len(c_anchor_nodes),
    }
    threshold_map = thresholds.model_dump()
    checks = {
        "component_nodes": metrics["component_nodes"] >= threshold_map["min_component_nodes"],
        "component_edges": metrics["component_edges"] >= threshold_map["min_component_edges"],
        "component_node_fraction": metrics["component_node_fraction"] >= threshold_map["min_component_node_fraction"],
        "relation_types": metrics["relation_types"] >= threshold_map["min_relation_types"],
        "b_anchor_nodes": metrics["b_anchor_nodes"] >= threshold_map["min_b_anchor_nodes"],
    }
    graph = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "modules_used": ["B", "C", "E"],
        "modules_ignored": ["D"],
        "paper": paper,
        "source": {"sourceid": source_id, "sourcefilepath": source_text, "sha256": source_sha256},
        "nodes": sorted(nodes.values(), key=lambda row: row["id"]),
        "edges": sorted(edges, key=lambda row: row["id"]),
    }
    lcc_graph = {
        **{key: value for key, value in graph.items() if key not in {"nodes", "edges"}},
        "component_id": largest.get("component_id", ""),
        "nodes": [node for node in graph["nodes"] if node["id"] in largest_node_ids],
        "edges": largest_edges,
    }
    assessment = {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": "accepted" if all(checks.values()) else "rejected",
        "passed": all(checks.values()),
        "paper": paper,
        "source": graph["source"],
        "modules_used": ["B", "C", "E"],
        "modules_ignored": ["D"],
        "counts": {
            "raw_relations": len(raw_relations),
            "deduplicated_nodes": total_nodes,
            "deduplicated_edges": len(edges),
            "connected_components": len(component_rows),
        },
        "thresholds": threshold_map,
        "metrics": metrics,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "largest_component": {
            "component_id": largest.get("component_id", ""),
            "node_count": metrics["component_nodes"],
            "edge_count": metrics["component_edges"],
            "relation_types": relation_types,
            "b_anchor_node_names": b_anchor_nodes,
            "c_anchor_node_names": c_anchor_nodes,
            "hub_nodes": hubs,
        },
        "components": [
            {key: value for key, value in component.items() if key not in {"node_ids", "edge_ids"}}
            for component in component_rows
        ],
    }
    return ScreenResult(assessment=assessment, graph=graph, largest_component_graph=lcc_graph)


def _paper_identity(value: dict[str, Any]) -> dict[str, str]:
    module = value.get(MODULE_A) or {}
    paper = module.get("Paper") if isinstance(module, dict) else {}
    paper = paper if isinstance(paper, dict) else {}
    return {
        "title": _text(paper.get("title")),
        "doi": _meaningful_text(paper.get("doi")),
        "arxiv_id": _meaningful_text(paper.get("arxiv_id")),
        "publication_year": _meaningful_text(paper.get("pub_year")),
    }


def _aliases(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        _normalize_name(alias): _normalize_name(canonical)
        for alias, canonical in value.items()
        if _text(alias) and _text(canonical)
    }


def _b_labels(value: Any, aliases: dict[str, str]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    if not isinstance(value, dict):
        return output
    for category, records in value.items():
        field = B_NAME_FIELDS.get(str(category))
        if not field:
            continue
        items = records if isinstance(records, list) else [records]
        for record in items:
            if not isinstance(record, dict):
                continue
            name = _text(record.get(field))
            if name and name.casefold() not in MISSING:
                output[_canonical(name, aliases)].add(str(category))
    return output


def _c_labels(value: Any, aliases: dict[str, str]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    if not isinstance(value, dict):
        return output
    for category, content in value.items():
        for phrase in _scalar_phrases(content):
            normalized = _canonical(phrase, aliases)
            if len(normalized) >= 4 and phrase.casefold() not in MISSING:
                output[normalized].add(str(category))
    return output


def _scalar_phrases(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _scalar_phrases(item)
    elif isinstance(value, list):
        for item in value:
            yield from _scalar_phrases(item)
    elif isinstance(value, str) and value.strip():
        yield value.strip()


def _canonical(value: Any, aliases: dict[str, str]) -> str:
    normalized = _normalize_name(value)
    visited = set()
    while normalized in aliases and normalized not in visited:
        visited.add(normalized)
        normalized = aliases[normalized]
    return normalized


def _normalize_name(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _normalize_relation(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    return re.sub(r"[^A-Z0-9]+", "_", normalized).strip("_")


def _node_id(canonical: str) -> str:
    return f"node_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


def _upsert_node(
    nodes: dict[str, dict[str, Any]],
    node_id: str,
    canonical: str,
    raw_name: str,
    raw_type: Any,
    b_labels: dict[str, set[str]],
    c_labels: dict[str, set[str]],
) -> None:
    node = nodes.setdefault(
        node_id,
        {
            "id": node_id,
            "name": raw_name,
            "canonical_name": canonical,
            "aliases": [],
            "types": [],
            "b_categories": sorted(b_labels.get(canonical, set())),
            "c_categories": sorted(c_labels.get(canonical, set())),
        },
    )
    if raw_name != node["name"] and raw_name not in node["aliases"]:
        node["aliases"].append(raw_name)
        node["aliases"].sort(key=str.casefold)
    node_type = _text(raw_type)
    if node_type and node_type not in node["types"]:
        node["types"].append(node_type)
        node["types"].sort(key=str.casefold)


def _components(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    incident: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])
        incident[edge["source"]].append(edge["id"])
        incident[edge["target"]].append(edge["id"])
    unseen = set(nodes)
    output = []
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        component_nodes = []
        while queue:
            current = queue.popleft()
            component_nodes.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        node_set = set(component_nodes)
        component_edges = sorted({edge_id for node_id in node_set for edge_id in incident[node_id]})
        output.append(
            {
                "node_count": len(component_nodes),
                "edge_count": len(component_edges),
                "node_ids": sorted(component_nodes),
                "edge_ids": component_edges,
            }
        )
    output.sort(key=lambda row: (-row["node_count"], -row["edge_count"], row["node_ids"][0]))
    return output


def _empty_component() -> dict[str, Any]:
    return {"component_id": "", "rank": 0, "node_count": 0, "edge_count": 0, "node_ids": [], "edge_ids": []}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _meaningful_text(value: Any) -> str:
    text = _text(value)
    return "" if text.casefold() in MISSING else text
