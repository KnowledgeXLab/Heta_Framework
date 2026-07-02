import asyncio
import os
import shutil
from pathlib import Path

from heta_framework.common.models import EmbeddingModel, LanguageModel
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore
from heta_framework.kb import (
    DocumentParserRegistry,
    EmbedChunks,
    IndexVectors,
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
    # 0. 准备一个干净 workspace；真实服务里通常换成你的业务目录或对象存储。
    workspace = Path("heta-demo-vector")
    shutil.rmtree(workspace, ignore_errors=True)

    # 1. Stores：ObjectStore 管原始文件和中间产物，VectorStore 管向量索引。
    objects = LocalObjectStore(workspace / "objects")
    vectors = InMemoryVectorStore()

    # 2. Models：Heta 的 model client 通过 LiteLLM 调用常见外部模型。
    #    运行前设置：export OPENAI_API_KEY=...
    language = LanguageModel(
        model_name=os.getenv("HETA_LLM_MODEL", "openai/gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    embedding = EmbeddingModel(
        model_name=os.getenv("HETA_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # 3. 输入文档：这里写入一个 txt；PDF/HTML/Office 可以换成对应 parser。
    await objects.put(
        "raw/heta.txt",
        (
            "Heta builds KnowledgeBase objects from Recipe definitions. "
            "Vector search retrieves chunks by semantic similarity."
        ).encode("utf-8"),
    )

    # 4. Recipe：声明 parser、model、store，以及要执行的 build steps。
    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
        models=KnowledgeModels(language=language, embedding=embedding),
        stores=KnowledgeStores(objects=objects, vector=vectors),
        steps=(
            ParseDocuments(),
            SplitDocuments(SplitDocumentsConfig(encoding_name="unicode")),
            EmbedChunks(),
            IndexVectors(),
        ),
    )

    # 5. Build + query：用真实 embedding API 建向量索引。
    #    generate_answer=True 时，query engine 会再调用 LLM 生成 answer。
    kb = await KnowledgeBase.create(recipe=recipe, name="home-vector")
    _raise_if_build_failed(kb)
    response = await kb.query(
        "How does Heta build a knowledge base?",
        mode="vector_search",
        top_k=1,
        options={"generate_answer": True},
    )

    print(response.answer)
    print(response.results[0].text)

    await language.aclose()
    await embedding.aclose()
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
