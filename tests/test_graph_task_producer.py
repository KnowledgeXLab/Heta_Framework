import asyncio
import hashlib
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from heta_framework.common.stores import LocalObjectStore
from heta_framework.generation import (
    HetaGenTaskProducerGraph,
    HetaGenTaskProducerGraphConfig,
    ObjectInput,
    TaskProducerProtocol,
    TaskRequest,
)
from heta_framework.generation.graph import GraphInputError
from heta_framework.generation.graph.inputs import stage_snapshot
from heta_framework.kb import KnowledgeBase
from heta_framework.kb.chunking.types import ParsedChunk
from heta_framework.kb.components import KnowledgeStores
from heta_framework.kb.graphing.types import ExtractedEntity, ExtractedRelation
from heta_framework.kb.parsing.types import (
    ParsedDocument,
    ParsedPage,
    ParsedSource,
    ParsedTextContent,
)
from heta_framework.kb.recipe import KnowledgeRecipe
from heta_framework.kb.search import QueryEngineRegistry, QueryResponse, QueryResult
from heta_framework.kb.state import RecipeRunRecord
from heta_framework.kb.steps import StepCapabilities, StepRequirements, store_ref


class FixtureSearch:
    mode = "fixture"
    required_assets = frozenset()

    def __init__(self, results):
        self.results = results

    async def query(self, request, context):
        assert request.options["generate_answer"] is False
        return QueryResponse(mode=self.mode, results=self.results)


async def make_kb(tmp_path, *, source_id="doc-1", empty=False, entity_type="METHOD"):
    source_store = LocalObjectStore(tmp_path / "sources")
    text = "Method A uses dataset B.\nThe experiment evaluates their relationship.\n"
    source = ParsedSource(
        "papers/paper.txt",
        "Scientific experiment",
        "txt",
        hashlib.sha256(text.encode()).hexdigest(),
    )
    document = ParsedDocument(
        "doc-1", source, [ParsedPage(0, text)], ParsedTextContent(text, "text/plain")
    )
    chunk = ParsedChunk("chunk-1", "doc-1", source, 0, 0, text, 0, 10)
    entities = [
        ExtractedEntity(
            "a",
            "chunk-1",
            "doc-1",
            "Method A",
            entity_type,
            None,
            "A scientific method",
            {},
            ("chunk-1",),
        ),
        ExtractedEntity(
            "b",
            "chunk-1",
            "doc-1",
            "dataset B",
            "DATASET",
            None,
            "Experiment data",
            {},
            ("chunk-1",),
        ),
    ]
    relation = ExtractedRelation(
        "r",
        "chunk-1",
        "doc-1",
        "a",
        "b",
        "Method A",
        "dataset B",
        "USES",
        "uses",
        "Uses experimental data",
        {},
        ("chunk-1",),
    )
    groups = {
        "parsed_document_keys": [document],
        "chunk_keys": [chunk],
        "entity_keys": entities,
        "relation_keys": [relation],
    }
    artifacts = {}
    for artifact, values in groups.items():
        artifacts[artifact] = []
        for index, value in enumerate(values):
            key = f"build/{artifact}/{index}.json"
            await source_store.put(key, value.to_json_bytes())
            artifacts[artifact].append(key)
    # Exercise named source-store resolution, separate from the output store.
    step = SimpleNamespace(
        name="fixture_artifacts",
        requirements=StepRequirements(components=frozenset({store_ref("objects", "archive")})),
        capabilities=StepCapabilities(artifacts=frozenset(groups)),
    )
    recipe = KnowledgeRecipe(
        stores=KnowledgeStores(named={"objects.archive": source_store}), steps=(step,)
    )
    results = (
        ()
        if empty
        else (QueryResult("hit", "Method A uses dataset B.", source={"document_id": source_id}),)
    )
    return KnowledgeBase(
        "test-kb",
        None,
        recipe,
        RecipeRunRecord("build-1", "succeeded", "now", "now", (), artifacts),
        "now",
        "now",
        query_engines=QueryEngineRegistry((FixtureSearch(results),)),
    )


def config(**kwargs):
    return HetaGenTaskProducerGraphConfig(query_mode="fixture", **kwargs)


def test_graph_protocol_and_declarations():
    producer = HetaGenTaskProducerGraph(config())
    assert isinstance(producer, TaskProducerProtocol)
    assert producer.requirements.inputs[0].name == "query"
    assert {
        slot.name
        for slot in HetaGenTaskProducerGraph(config(input_mode="objects")).requirements.inputs
    } == {"kg", "text"}
    with pytest.raises(ValueError):
        config(max_documents=0)


def test_kb_snapshot_separate_stores_and_object_reuse(tmp_path):
    async def run():
        kb = await make_kb(tmp_path)
        store = LocalObjectStore(tmp_path / "output")
        producer = HetaGenTaskProducerGraph(config())
        prepared = await producer.prepare_inputs(
            TaskRequest("prepared", {"query": "scientific methods"}),
            object_store=store,
            knowledge_base=kb,
        )
        assert prepared.document_count == 1
        snapshot = json.loads(await store.get(prepared.manifest_key))
        assert snapshot["provenance"]["run_id"] == "build-1"
        assert {read["store"] for read in snapshot["provenance"]["objects_read"]} == {
            "stores.objects.archive"
        }
        graph = json.loads(await store.get(snapshot["documents"][0]["kg"]["key"]))
        assert graph["B_Textually_Mentioned_Entities"]["Methods"][0]["evidence"][0][
            "source_lines"
        ] == [1, 2]
        other = HetaGenTaskProducerGraph(config(input_mode="objects"))
        reused = await other.prepare_inputs(
            TaskRequest(
                "reused",
                {
                    "kg": ObjectInput(prefix="prepared/inputs/kg/"),
                    "text": ObjectInput(prefix="prepared/inputs/text/"),
                },
            ),
            object_store=store,
        )
        again = json.loads(await store.get(reused.manifest_key))
        assert again["documents"][0]["kg"]["sha256"] == snapshot["documents"][0]["kg"]["sha256"]
        first_rows = await stage_snapshot(prepared, store, tmp_path / "staged-first")
        reused_rows = await stage_snapshot(reused, store, tmp_path / "staged-reused")
        assert first_rows[0]["paper_id"] == reused_rows[0]["paper_id"]
        await store.put(snapshot["documents"][0]["text"]["key"], b"changed")
        with pytest.raises(GraphInputError, match="snapshot changed"):
            await stage_snapshot(prepared, store, tmp_path / "staged")

    asyncio.run(run())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_id": "missing"}, "unknown document"),
        ({"entity_type": "UNSUPPORTED"}, "unsupported entity type"),
    ],
)
def test_kb_invalid_sources_and_types_fail_before_snapshot(tmp_path, kwargs, message):
    async def run():
        kb = await make_kb(tmp_path, **kwargs)
        store = LocalObjectStore(tmp_path / "out")
        with pytest.raises(GraphInputError, match=message):
            await HetaGenTaskProducerGraph(config()).prepare_inputs(
                TaskRequest("run", {"query": "methods"}), object_store=store, knowledge_base=kb
            )
        assert not await store.list()

    asyncio.run(run())


def test_no_hits_produces_zero_tasks_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def run():
        kb = await make_kb(tmp_path, empty=True)
        store = LocalObjectStore(tmp_path / "output")
        result = await HetaGenTaskProducerGraph(config(env_file=tmp_path / "absent")).produce(
            TaskRequest("run", {"query": "no matches"}), object_store=store, knowledge_base=kb
        )
        assert result.status == "succeeded"
        assert json.loads(await store.get(result.tasks_manifest_key))["tasks"] == []
        with pytest.raises(FileExistsError):
            await HetaGenTaskProducerGraph(config()).produce(
                TaskRequest("run", {"query": "again"}), object_store=store, knowledge_base=kb
            )

    asyncio.run(run())


def test_real_native_designer_rejects_small_graph_without_model_calls(tmp_path, monkeypatch):
    pytest.importorskip("data_designer")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-unused")

    async def run():
        kb = await make_kb(tmp_path)
        store = LocalObjectStore(tmp_path / "output")
        producer = HetaGenTaskProducerGraph(
            config(api_base_url="http://127.0.0.1:1/v1", env_file=tmp_path / "absent")
        )
        result = await producer.produce(
            TaskRequest("run", {"query": "methods"}), object_store=store, knowledge_base=kb
        )
        report = json.loads(await store.get(result.report_key))
        assert result.status == "succeeded", report
        assert report["native"]["stages"]["screen"] == "completed_empty"
        assert report["native"]["stages"]["review"] == "skipped_empty_upstream"
        assert report["native"]["rejections"][0]["detail"]["count"] == 1
        assert report["delivered_task_count"] == 0

    asyncio.run(run())


def test_vendored_assets_match_upstream_hashes():
    import heta_framework.generation.graph._native as native

    base = Path(native.__file__).parent
    for line in (base / "assets.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        assert hashlib.sha256((base / "assets" / relative).read_bytes()).hexdigest() == expected
    assert (base / "LICENSE").exists() and (base / "NOTICE").exists()


def test_real_designer_model_boundary_and_semantic_rejection(tmp_path, monkeypatch):
    """A local OpenAI fixture exercises the real native review column and parser."""
    pytest.importorskip("data_designer")
    from heta_framework.generation.graph._native.domain.schemas import (
        REQUIRED_EVIDENCE_INVENTORY,
        REQUIRED_FEASIBILITY_CHECKS,
    )

    review = {
        "paper_summary": "Offline fixture, insufficient scientific details",
        "component_summary": "Connected graph fixture",
        "suitable_for_task_generation": False,
        "suitability_score": 0.1,
        "evidence_inventory": {key: [] for key in REQUIRED_EVIDENCE_INVENTORY},
        "constraint_checks": dict.fromkeys(REQUIRED_FEASIBILITY_CHECKS, False),
        "computational_affordances": [],
        "blockers": ["fixture lacks equations"],
    }
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(
                {
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(review)},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture-key")

    async def run():
        store = LocalObjectStore(tmp_path / "output")
        text = "\n".join(f"Method {i} connects to method {(i + 1) % 20}." for i in range(20))
        evidence = [{"source_lines": [1, 20], "quote": text}]
        graph = {
            "A_Meta_Factual_Entities": {"Paper": {"title": "Offline fixture"}},
            "B_Textually_Mentioned_Entities": {
                "Methods": [{"name": f"Method {i}", "evidence": evidence} for i in range(20)]
            },
            "E_Relations": [
                {
                    "head": f"Method {i}",
                    "tail": f"Method {(i + 1) % 20}",
                    "relation": ("USES", "EVALUATES", "COMPARES")[i % 3],
                    "evidence": evidence,
                }
                for i in range(20)
            ],
        }
        await store.put("source/paper.json", json.dumps(graph).encode())
        await store.put("source/paper.txt", text.encode())
        producer = HetaGenTaskProducerGraph(
            config(
                input_mode="objects",
                model="fixture-model",
                api_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                env_file=tmp_path / "absent",
            )
        )
        result = await producer.produce(
            TaskRequest(
                "run",
                {
                    "kg": ObjectInput(keys=("source/paper.json",)),
                    "text": ObjectInput(keys=("source/paper.txt",)),
                },
            ),
            object_store=store,
        )
        report = json.loads(await store.get(result.report_key))
        assert result.status == "succeeded", report
        assert report["native"]["stages"]["screen"] == "completed"
        assert report["native"]["stages"]["review"] == "completed_empty"
        assert report["native"]["rejections"][1]["kind"] == "rejected_semantic"
        assert calls and all(call["model"] == "fixture-model" for call in calls)
        assert "Offline fixture" in json.dumps(calls)
        assert await store.exists(report["designer_evidence_key"])

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def ready_package(tmp_path):
    from heta_framework.generation.graph._native.domain.validation_runtime import write_report

    package = tmp_path / "package"
    package.mkdir()
    report = {
        "terminal_bench_ready": True,
        "static_passed": True,
        "data_materialized": True,
        "deterministic": True,
        "harbor": {},
    }
    for agent, reward, count in (("oracle", 1.0, 2), ("nop", 0.0, 1)):
        report["harbor"][agent] = {
            "passed": True,
            "runs": [
                {"returncode": 0, "completed": True, "reward": reward, "error": ""}
                for _ in range(count)
            ],
        }
    report_path = write_report(package, report)
    row = {
        "package_dir": str(package),
        "validation_report_path": str(report_path),
        "terminal_bench_ready": True,
        "static_passed": True,
        "materialize_ok": True,
        "harbor_passed": True,
        "deterministic": True,
        "oracle_rewards": [1.0, 1.0],
        "nop_reward": 0.0,
    }
    return package, report, row


@pytest.mark.parametrize(
    "mutation",
    ["missing_report", "row_gate", "report_gate", "missing_trial", "wrong_reward", "escape"],
)
def test_publication_requires_matching_row_and_trial_evidence(tmp_path, mutation):
    pytest.importorskip("data_designer")
    from heta_framework.generation.graph.designer import accepted_package

    package, report, row = ready_package(tmp_path)
    assert accepted_package(row, tmp_path, 2) == package
    if mutation == "missing_report":
        Path(row["validation_report_path"]).unlink()
    elif mutation == "row_gate":
        row["static_passed"] = False
    elif mutation == "report_gate":
        report["data_materialized"] = False
    elif mutation == "missing_trial":
        report["harbor"]["oracle"]["runs"].pop()
    elif mutation == "wrong_reward":
        report["harbor"]["nop"]["runs"][0]["reward"] = 1.0
    elif mutation == "escape":
        row["package_dir"] = str(tmp_path.parent)
    if mutation != "missing_report":
        Path(row["validation_report_path"]).write_text(json.dumps(report))
    with pytest.raises((ValueError, FileNotFoundError)):
        accepted_package(row, tmp_path, 2)


def test_explicit_entity_mapping_preserves_kb_semantics(tmp_path):
    async def run():
        kb = await make_kb(tmp_path, entity_type="CUSTOM_METHOD")
        store = LocalObjectStore(tmp_path / "out")
        producer = HetaGenTaskProducerGraph(
            config(entity_type_mapping={"CUSTOM_METHOD": "Methods"})
        )
        result = await producer.prepare_inputs(
            TaskRequest("run", {"query": "methods"}), object_store=store, knowledge_base=kb
        )
        manifest = json.loads(await store.get(result.manifest_key))
        graph = json.loads(await store.get(manifest["documents"][0]["kg"]["key"]))
        assert graph["B_Textually_Mentioned_Entities"]["Methods"][0]["name"] == "Method A"
        with pytest.raises(TypeError):
            producer.config.entity_type_mapping["OTHER"] = "Tasks"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_chunk", "missing source chunk"),
        ("cross_document", "cross-document endpoint"),
        ("wrong_endpoint_name", "endpoint name mismatch"),
        ("missing_artifact", "missing or empty KB artifact"),
        ("bad_text", "absent or ambiguous"),
    ],
)
def test_existing_kb_evidence_is_validated_before_publishing_inputs(tmp_path, mutation, message):
    async def run():
        kb = await make_kb(tmp_path)
        source = kb.recipe.stores.named["objects.archive"]
        artifact = "entity_keys" if mutation == "missing_chunk" else "relation_keys"
        if mutation == "missing_artifact":
            del kb.run_record.artifacts["relation_keys"]
        elif mutation == "bad_text":
            key = kb.run_record.artifacts["chunk_keys"][0]
            item = json.loads(await source.get(key))
            item["text"] = "Not present in the paper"
            await source.put(key, json.dumps(item).encode())
        else:
            key = kb.run_record.artifacts[artifact][0]
            item = json.loads(await source.get(key))
            if mutation == "missing_chunk":
                item["source_chunk_ids"] = ["absent"]
            elif mutation == "cross_document":
                item["target_entity_id"] = "other-document-entity"
            else:
                item["target_entity_name"] = "invented name"
            await source.put(key, json.dumps(item).encode())
        store = LocalObjectStore(tmp_path / "output")
        with pytest.raises(GraphInputError, match=message):
            await HetaGenTaskProducerGraph(config()).prepare_inputs(
                TaskRequest("run", {"query": "methods"}), object_store=store, knowledge_base=kb
            )
        assert await store.list() == []

    asyncio.run(run())


@pytest.mark.parametrize("invalid_certificate", [False, True])
def test_publication_preserves_native_generator_and_partial_outcomes(
    tmp_path, monkeypatch, invalid_certificate
):
    """Injected execution tests publication; it is not evidence of a live valid task."""
    pytest.importorskip("data_designer")
    from heta_framework.generation.graph import designer

    def execute(rows, root, config, provider):
        root.mkdir(parents=True)
        package, _, row = ready_package(root)
        for relative, content in {
            "task.toml": 'version = "1.0"\n',
            "instruction.md": "Scientific fixture",
            "generation_data.py": "print('fixture')\n",
            "generation/Dockerfile": "FROM scratch\n",
            "execution.md": "Data generation fixture",
            "environment/data/public.csv": "input\n1\n",
            "tests/data/hidden.csv": "expected\n2\n",
            "solution/solve.sh": "#!/bin/sh\nexit 0\n",
        }.items():
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        failed_row = (
            {**row, "paper_id": "failed", "validation_report_path": str(root / "missing.json")}
            if invalid_certificate
            else {"terminal_bench_ready": False, "paper_id": "failed"}
        )
        return {
            "complete": True,
            "stages": {"validate_runtime": "completed"},
            "rows": [
                {**row, "paper_id": rows[0]["paper_id"], "tier": "complex"},
                failed_row,
            ],
            "oracle_repeats": 2,
            "rejections": [],
            "stage_row_counts": {},
        }

    monkeypatch.setattr(designer, "execute", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture-key")

    async def run():
        kb = await make_kb(tmp_path)
        store = LocalObjectStore(tmp_path / "output")
        result = await HetaGenTaskProducerGraph(
            config(api_base_url="http://127.0.0.1:1/v1")
        ).produce(TaskRequest("run", {"query": "methods"}), object_store=store, knowledge_base=kb)
        assert result.status == "partial"
        tasks = json.loads(await store.get(result.tasks_manifest_key))["tasks"]
        assert len(tasks) == 1
        assert {
            "generation_data.py",
            "generation/Dockerfile",
            "execution.md",
            "environment/data/public.csv",
            "tests/data/hidden.csv",
        } <= {file["path"] for file in tasks[0]["files"]}
        assert all(not file["path"].startswith("authoring/") for file in tasks[0]["files"])
        assert json.loads(await store.get(tasks[0]["evidence_keys"][0]))["terminal_bench_ready"]

    asyncio.run(run())


def test_native_materialization_isolates_same_slug_across_documents(tmp_path, monkeypatch):
    pytest.importorskip("data_designer")
    import pandas as pd

    from heta_framework.generation.graph._native.dd import workflow

    stage = tmp_path / "stage"
    (stage / "parquet-files").mkdir(parents=True)
    rows = [
        {
            "paper_id": paper_id,
            "package_ok": True,
            "tier": "complex",
            "spec": json.dumps(
                {
                    "marker": paper_id,
                    "files": {"task.toml": '[task]\nname = "terminal-bench-science/same-slug"\n'},
                }
            ),
            "implementation": "{}",
        }
        for paper_id in ("paper-a", "paper-b")
    ]
    pd.DataFrame(rows).to_parquet(stage / "parquet-files" / "data.parquet", index=False)
    monkeypatch.setattr(
        workflow.columns,
        "render_package_files",
        lambda spec, implementation: {"instruction.md": spec["marker"]},
    )
    callback = workflow.make_materialize_callback(
        version="test", packages_root=tmp_path / "packages", paper_namespaced=True
    )
    output = pd.read_parquet(callback(stage) / "data.parquet")
    packages = [Path(value) for value in output["package_dir"]]
    assert len(set(packages)) == 2
    assert all(path.name == "same-slug" for path in packages)
    assert [(path / "instruction.md").read_text() for path in packages] == ["paper-a", "paper-b"]


@pytest.mark.parametrize("bad_hash", [False, True])
def test_wiki_origin_hash_is_checked_against_original_document(tmp_path, bad_hash):
    async def run():
        kb = await make_kb(tmp_path)
        source = kb.recipe.stores.named["objects.archive"]
        document = ParsedDocument.from_json(
            await source.get(kb.run_record.artifacts["parsed_document_keys"][0])
        )
        engine = FixtureSearch(
            (
                QueryResult(
                    "wiki-hit",
                    "Heat methods",
                    source={
                        "document_id": "wiki_page_1",
                        "object_type": "wiki",
                        "object_key": "wiki/page.md",
                        "origin": {"object_key": document.source.key},
                    },
                    metadata={
                        "origin_source_content_sha256": "0" * 64
                        if bad_hash
                        else document.source.content_sha256
                    },
                ),
            )
        )
        kb = replace(kb, query_engines=QueryEngineRegistry((engine,)))
        store = LocalObjectStore(tmp_path / "output")
        producer = HetaGenTaskProducerGraph(config())
        if bad_hash:
            with pytest.raises(GraphInputError, match="origin content hash"):
                await producer.prepare_inputs(
                    TaskRequest("run", {"query": "heat"}), object_store=store, knowledge_base=kb
                )
            assert await store.list() == []
        else:
            prepared = await producer.prepare_inputs(
                TaskRequest("run", {"query": "heat"}), object_store=store, knowledge_base=kb
            )
            manifest = json.loads(await store.get(prepared.manifest_key))
            assert manifest["documents"][0]["document_id"] == document.document_id

    asyncio.run(run())


def test_user_scientific_constraints_reach_every_authoring_stage(tmp_path):
    pytest.importorskip("data_designer")
    from heta_framework.generation.graph._native.config import Settings
    from heta_framework.generation.graph._native.dd import columns

    guidance = "Use an isotropic positive cubic saturation; the degree-three library has 20 terms."
    settings = Settings(generation_guidance=guidance)
    paper = tmp_path / "paper.txt"
    paper.write_text("Scientific source evidence.\n")
    kg = tmp_path / "kg.json"
    kg.write_text(
        json.dumps({"A_Meta_Factual_Entities": {"Paper": {"title": "Source"}}, "E_Relations": []})
    )
    spec = {
        "package_name": "test",
        "task_spec": {},
        "runtime_spec": {},
        "data_contract": {},
        "evaluation_topology": {},
        "difficulty_contract": {},
        "files": {"instruction.md": "Instruction", "task.toml": "version='1.0'"},
    }
    row = {
        "kg_path": str(kg),
        "paper_text_path": str(paper),
        "semantic_review": "{}",
        "tier": "complex",
        "spec": json.dumps(spec),
    }
    for payload in (
        columns._candidate_payload(row, settings),
        columns._spec_payload(row, settings),
        columns._implementation_payload(row, settings),
        columns._implementation_check_payload(spec, {"files": {}}, settings),
    ):
        assert payload["generation_guidance"] == guidance
        if "source_evidence" in payload:
            assert payload["source_evidence"]["excerpt"] == paper.read_text()
