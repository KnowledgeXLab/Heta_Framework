import asyncio

import pytest

from graph_query_matrix import run_matrix


def test_all_builtin_queries_resolve_real_built_kb_inputs(tmp_path):
    pytest.importorskip("sqlalchemy")
    report = asyncio.run(run_matrix(tmp_path / "matrix"))
    assert not report["uncovered_modes"], report
    assert all(kb["status"] == "succeeded" for kb in report["knowledge_bases"].values()), report
    assert all(query["status"] == "passed" for query in report["queries"]), report


def test_all_builtin_queries_complete_native_producer_handoff(tmp_path):
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("data_designer")
    report = asyncio.run(run_matrix(tmp_path / "matrix", run_producer=True))
    assert report["status"] == "passed", report
    assert len(report["queries"]) == len(report["registered_modes"])
    assert all(query["production_status"] == "succeeded" for query in report["queries"])
    assert all(query["task_count"] == 0 for query in report["queries"])
