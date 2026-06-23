import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import (  # noqa: E402
    GraphEdge,
    GraphNode,
    GraphStoreProtocol,
    InMemoryGraphStore,
)


def test_in_memory_graph_store_satisfies_protocol():
    assert isinstance(InMemoryGraphStore(), GraphStoreProtocol)


def test_graph_node_validates_required_fields():
    with pytest.raises(ValueError, match="id"):
        GraphNode(id="")
    with pytest.raises(ValueError, match="labels"):
        GraphNode(id="entity_1", labels=())


def test_graph_edge_validates_required_fields():
    with pytest.raises(ValueError, match="source_id"):
        GraphEdge(id="relation_1", source_id="", target_id="entity_2", type="contains")
    with pytest.raises(ValueError, match="different"):
        GraphEdge(id="relation_1", source_id="entity_1", target_id="entity_1", type="contains")


def test_in_memory_graph_store_upserts_reads_and_deletes():
    store = InMemoryGraphStore()

    async def run():
        await store.upsert_nodes(
            [
                GraphNode(id="entity_a", labels=("Entity", "City"), properties={"name": "上海市"}),
                GraphNode(
                    id="entity_b",
                    labels=("Entity", "District"),
                    properties={"name": "徐汇区"},
                ),
            ]
        )
        await store.upsert_edges(
            [
                GraphEdge(
                    id="relation_a",
                    source_id="entity_a",
                    target_id="entity_b",
                    type="包含行政区",
                    properties={"type": "空间关系"},
                )
            ]
        )
        first_count = await store.count_nodes(), await store.count_edges()
        node = await store.get_node("entity_a")
        edge = await store.get_edge("relation_a")
        await store.upsert_nodes(
            [GraphNode(id="entity_a", labels=("Entity", "Municipality"), properties={})]
        )
        updated = await store.get_node("entity_a")
        await store.delete_edges(["relation_a"])
        after_edge_delete = await store.count_edges()
        await store.upsert_edges(
            [GraphEdge(id="relation_b", source_id="entity_a", target_id="entity_b", type="管辖")]
        )
        await store.delete_nodes(["entity_b"])
        final_count = await store.count_nodes(), await store.count_edges()
        return first_count, node, edge, updated, after_edge_delete, final_count

    first_count, node, edge, updated, after_edge_delete, final_count = asyncio.run(run())

    assert first_count == (2, 1)
    assert node is not None
    assert node.properties["name"] == "上海市"
    assert edge is not None
    assert edge.type == "包含行政区"
    assert updated is not None
    assert updated.labels == ("Entity", "Municipality")
    assert after_edge_delete == 0
    assert final_count == (1, 0)
