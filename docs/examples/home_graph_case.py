import asyncio
import os
import shutil
from pathlib import Path

from heta_framework.common.models import EmbeddingModel, LanguageModel
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore, SQLStore
from heta_framework.kb import (
    DocumentParserRegistry,
    HetaGraphProcedure,
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    ParseDocuments,
    SplitDocuments,
    SplitDocumentsConfig,
    TextParser,
)


async def main() -> None:
    # 0. 准备一个干净 workspace。
    workspace = Path("heta-demo-graph")
    shutil.rmtree(workspace, ignore_errors=True)

    # 1. Stores：ObjectStore 管文件产物，SQLStore 落图谱 facts，VectorStore 支持图谱召回。
    objects = LocalObjectStore(workspace / "objects")
    sql = SQLStore(f"sqlite:///{workspace / 'graph.db'}")
    vectors = InMemoryVectorStore()

    # 2. Models：LLM 抽取 entity/relation，embedding model 为图谱 facts 建向量索引。
    #    运行前设置：export OPENAI_API_KEY=...
    language = LanguageModel(
        model_name=os.getenv("HETA_LLM_MODEL", "openai/gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    embedding = EmbeddingModel(
        model_name=os.getenv("HETA_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # 3. 输入文档：Heta graph procedure 会从 chunk 中抽取 entity/relation。
    await objects.put(
        "raw/heta.txt",
        (
            "Heta builds knowledge bases from recipes. "
            "A KnowledgeBase is created by running Recipe steps."
        ).encode("utf-8"),
    )

    # 4. Recipe：HetaGraphProcedure 会展开为 extract entities、extract relations、build graph。
    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
        models=KnowledgeModels(language=language, embedding=embedding),
        stores=KnowledgeStores(objects=objects, sql=sql, vector=vectors),
        steps=(
            ParseDocuments(),
            SplitDocuments(SplitDocumentsConfig(encoding_name="unicode")),
            *HetaGraphProcedure.build(deduplicate=False).steps(),
        ),
    )

    # 5. Build + query：BuildGraph 会解锁 heta_graph_search。
    kb = await KnowledgeBase.create(recipe=recipe, name="home-graph")
    _raise_if_build_failed(kb)
    response = await kb.query(
        "How does Heta create a KnowledgeBase?",
        mode="heta_graph_search",
        top_k=3,
        options={"generate_answer": True},
    )

    print(response.answer)
    print(response.results[0].kind, response.results[0].text)

    await language.aclose()
    await embedding.aclose()
    await sql.aclose()
    await vectors.aclose()
    await objects.aclose()


def _raise_if_build_failed(kb: KnowledgeBase) -> None:
    if kb.run_record.status == "succeeded":
        return
    failed_step = next(
        (record for record in reversed(kb.run_record.step_records) if record.status == "failed"),
        None,
    )
    if failed_step is None:
        raise RuntimeError(f"knowledge base build failed: {kb.run_record.status}")
    raise RuntimeError(f"{failed_step.step_name} failed: {failed_step.error}")


asyncio.run(main())
