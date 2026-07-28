import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import EmbeddingRequest, EmbeddingResult  # noqa: E402
from heta_framework.common.stores import (  # noqa: E402
    InMemoryTextIndexStore,
    InMemoryVectorStore,
    LocalObjectStore,
    TextQuery,
)
from heta_framework.common.stores.vector import VectorQuery  # noqa: E402
from heta_framework.kb import (  # noqa: E402
    BuildWikiPages,
    BuildWikiPagesConfig,
    EmbedChunks,
    EmbedChunksConfig,
    FullTextIndexNames,
    IndexFullText,
    IndexFullTextConfig,
    IndexVectors,
    IndexVectorsConfig,
    KnowledgeModels,
    KnowledgeRecipe,
    KnowledgeStores,
    SplitWikiPages,
    SplitWikiPagesConfig,
    WikiPageChunkLimitError,
)
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.parsing import (  # noqa: E402
    ParsedDocument,
    ParsedPage,
    ParsedSource,
    ParsedTextContent,
)
from heta_framework.kb.steps import StepCapabilities  # noqa: E402


class FakeContext:
    def __init__(self, components):
        self.components = components
        self.artifacts = {}

    def get_component(self, key):
        return self.components[key]

    def get_artifact(self, key):
        return self.artifacts[key]

    def set_artifact(self, key, value):
        self.artifacts[key] = value


class FakeEmbeddingModel:
    @property
    def model_name(self):
        return "fake-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[[float(len(text)), float(text.count("a")), 1.0] for text in request.texts],
            model_name=self.model_name,
        )

    async def embed_many(self, requests):
        return []


def _wiki_document(content: str) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc_biology",
        source=ParsedSource(
            key="raw/biology.md",
            name="biology.md",
            file_type="md",
            content_sha256="a" * 64,
        ),
        pages=[ParsedPage(page_index=0, text="flattened fallback")],
        original_content=ParsedTextContent(
            text=content,
            media_type="text/markdown",
        ),
    )


async def _build_page(object_store, context, content):
    document = _wiki_document(content)
    await object_store.put("parsed/doc_biology.json", document.to_json_bytes())
    context.set_artifact("parsed_document_keys", ("parsed/doc_biology.json",))
    await BuildWikiPages(
        BuildWikiPagesConfig(summary_mode="extractive", encoding_name="unicode")
    ).run(context)


def test_split_wiki_pages_declares_chunk_contract():
    step = SplitWikiPages()

    assert step.name == "split_wiki_pages"
    assert {ref.key for ref in step.requirements.components} == {"stores.objects"}
    assert step.requirements.artifacts == frozenset({"wiki_page_keys"})
    assert step.capabilities == StepCapabilities(
        artifacts=frozenset({"split_wiki_pages_result", "wiki_chunk_keys"})
    )


def test_split_wiki_pages_builds_heading_aware_chunks(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    context = FakeContext({"stores.objects": object_store})
    content = (
        "# Plant Cell Metabolism\n\n"
        "## Photosynthesis\n\nIntroductory pathway.\n\n"
        "### Chloroplast\n\nOrganelle context.\n\n"
        "#### Light Reactions\n\nATP and NADPH are produced.\n\n"
        "```markdown\n### Not a real heading\n```"
    )

    async def run():
        await _build_page(object_store, context, content)
        step = SplitWikiPages(
            SplitWikiPagesConfig(chunk_size=512, overlap=0, encoding_name="unicode")
        )
        await step.run(context)
        chunks = [
            ParsedChunk.from_json(await object_store.get(key))
            for key in context.artifacts["wiki_chunk_keys"]
        ]
        return chunks, step.cleanup_plan(context.artifacts)

    chunks, cleanup = asyncio.run(run())

    assert [chunk.heading_path for chunk in chunks] == [
        ("Photosynthesis",),
        ("Photosynthesis", "Chloroplast"),
        ("Photosynthesis", "Chloroplast", "Light Reactions"),
    ]
    assert all(chunk.source.file_type == "wiki" for chunk in chunks)
    assert all(chunk.source.key == "wiki/pages/1-plant-cell-metabolism.md" for chunk in chunks)
    assert all(chunk.origin_source == _wiki_document(content).source for chunk in chunks)
    assert all(
        ParsedChunk.from_json(chunk.to_json()).origin_source == chunk.origin_source
        for chunk in chunks
    )
    assert "Page: Plant Cell Metabolism" in chunks[-1].text
    assert "Section: Photosynthesis > Chloroplast > Light Reactions" in chunks[-1].text
    assert "### Not a real heading" in chunks[-1].text
    assert {target.value for target in cleanup.targets} == set(
        context.artifacts["wiki_chunk_keys"]
    )


def test_split_wiki_pages_splits_each_section_within_token_budget(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    context = FakeContext({"stores.objects": object_store})
    content = "# Cell\n\n## Membrane\n\n" + "lipid " * 80

    async def run():
        await _build_page(object_store, context, content)
        await SplitWikiPages(
            SplitWikiPagesConfig(chunk_size=400, overlap=10, encoding_name="unicode")
        ).run(context)
        return [
            ParsedChunk.from_json(await object_store.get(key))
            for key in context.artifacts["wiki_chunk_keys"]
        ]

    chunks = asyncio.run(run())

    assert len(chunks) > 1
    assert all(chunk.heading_path == ("Membrane",) for chunk in chunks)
    assert all(len(chunk.text) <= 400 for chunk in chunks)


def test_split_wiki_pages_handles_page_chunk_limit_atomically(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    context = FakeContext({"stores.objects": object_store})
    content = "# Cell\n\n## Membrane\n\n" + "lipid " * 80

    async def run(policy):
        await _build_page(object_store, context, content)
        await SplitWikiPages(
            SplitWikiPagesConfig(
                chunk_size=400,
                overlap=0,
                encoding_name="unicode",
                max_chunks_per_page=1,
                oversized_page_policy=policy,
            )
        ).run(context)

    with pytest.raises(WikiPageChunkLimitError, match="max_chunks_per_page"):
        asyncio.run(run("fail"))
    assert asyncio.run(object_store.list("wiki_chunks")) == []

    asyncio.run(run("skip"))
    result = context.artifacts["split_wiki_pages_result"]
    assert result.chunk_keys == ()
    assert result.skipped_page_keys == ("wiki/pages/1-cell.md",)
    assert result.issues[0].code == "wiki_page_chunk_limit_exceeded"


def test_wiki_static_recipe_connects_to_existing_index_steps():
    recipe = KnowledgeRecipe(
        models=KnowledgeModels(embedding=FakeEmbeddingModel()),
        stores=KnowledgeStores(objects=object(), vector=object(), text_index=object()),
        steps=(
            BuildWikiPages(BuildWikiPagesConfig(summary_mode="extractive")),
            SplitWikiPages(),
            EmbedChunks(EmbedChunksConfig(preset="wiki")),
            IndexVectors(IndexVectorsConfig(preset="wiki")),
            IndexFullText(
                IndexFullTextConfig(
                    preset="wiki",
                    index_names=FullTextIndexNames(chunk_text="wiki_test_full_text"),
                )
            ),
        ),
    )

    result = recipe.validate(initial_artifacts={"parsed_document_keys"})

    assert result.valid


def test_wiki_static_pipeline_runs_through_vector_and_full_text_indexes(tmp_path):
    object_store = LocalObjectStore(tmp_path / "objects")
    vector_store = InMemoryVectorStore()
    text_index_store = InMemoryTextIndexStore()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.vector": vector_store,
            "stores.text_index": text_index_store,
            "models.embedding": FakeEmbeddingModel(),
        }
    )
    content = (
        "# Plant Cell Metabolism\n\n"
        "## Photosynthesis\n\n"
        "Chloroplasts produce ATP and NADPH during the light reactions."
    )

    async def run():
        await _build_page(object_store, context, content)
        await SplitWikiPages(SplitWikiPagesConfig(encoding_name="unicode")).run(context)
        await EmbedChunks(EmbedChunksConfig(preset="wiki")).run(context)
        await IndexVectors(IndexVectorsConfig(preset="wiki")).run(context)
        await IndexFullText(IndexFullTextConfig(preset="wiki")).run(context)
        vector_hits = await vector_store.search(
            "wiki_chunks",
            VectorQuery(vector=[100.0, 4.0, 1.0], top_k=3),
        )
        text_hits = await text_index_store.search(
            "wiki_chunk_full_text",
            TextQuery(text="chloroplast ATP", top_k=3),
        )
        await text_index_store.aclose()
        return vector_hits, text_hits

    vector_hits, text_hits = asyncio.run(run())

    assert context.artifacts["build_wiki_pages_result"].page_count == 1
    assert context.artifacts["split_wiki_pages_result"].chunk_count == 1
    assert context.artifacts["embed_wiki_chunks_result"].chunk_count == 1
    assert context.artifacts["index_wiki_vectors_result"].indexed_count == 1
    assert context.artifacts["index_wiki_full_text_result"].indexed_count == 1
    assert vector_hits[0].metadata["heading_path"] == "Photosynthesis"
    assert text_hits[0].metadata["heading_path"] == "Photosynthesis"
