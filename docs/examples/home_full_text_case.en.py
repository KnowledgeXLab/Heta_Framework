import asyncio
import os
import shutil
from pathlib import Path

from heta_framework.common.models import LanguageModel
from heta_framework.common.stores import InMemoryTextIndexStore, LocalObjectStore
from heta_framework.kb import (
    DocumentParserRegistry,
    IndexFullText,
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
    # 0. Prepare a clean workspace.
    workspace = Path("heta-demo-full-text")
    shutil.rmtree(workspace, ignore_errors=True)

    # 1. Stores: ObjectStore keeps document artifacts; TextIndexStore keeps the full-text index.
    objects = LocalObjectStore(workspace / "objects")
    text_index = InMemoryTextIndexStore()

    # 2. Model: full_text_search does not require an LLM; the LLM is only for answer generation.
    #    Before running: export OPENAI_API_KEY=...
    language = LanguageModel(
        model_name=os.getenv("HETA_LLM_MODEL", "openai/gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # 3. Input document: full-text indexing is useful for exact terms, IDs, abbreviations, and keywords.
    await objects.put(
        "raw/heta.txt",
        (
            "Heta can add full-text search with IndexFullText. "
            "BM25-style retrieval is useful for exact terms and identifiers."
        ).encode("utf-8"),
    )

    # 4. Recipe: IndexFullText writes chunks into the text index.
    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
        models=KnowledgeModels(language=language),
        stores=KnowledgeStores(objects=objects, text_index=text_index),
        steps=(
            ParseDocuments(),
            SplitDocuments(SplitDocumentsConfig(encoding_name="unicode")),
            IndexFullText(),
        ),
    )

    # 5. Build + query: IndexFullText unlocks full_text_search.
    kb = await KnowledgeBase.create(recipe=recipe, name="home-full-text")
    _raise_if_build_failed(kb)
    response = await kb.query(
        "BM25 exact terms",
        mode="full_text_search",
        top_k=1,
        options={"generate_answer": True},
    )

    print(response.answer)
    print(response.results[0].text)

    await language.aclose()
    await text_index.aclose()
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
