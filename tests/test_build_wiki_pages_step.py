import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import ModelRequest, ModelResult  # noqa: E402
from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb import (  # noqa: E402
    BuildWikiPages,
    BuildWikiPagesConfig,
    WikiDocumentLimitError,
)
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


class FakeLanguageModel:
    def __init__(self, text="Short biology wiki summary."):
        self.text = text
        self.requests = []

    @property
    def model_name(self):
        return "fake-language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        return ModelResult(text=self.text, model_name=self.model_name)

    async def invoke_many(self, requests):
        return [await self.invoke(request) for request in requests]

    def stream(self, request):
        async def iterator():
            if False:
                yield None

        return iterator()


class FailingAfterFirstLanguageModel(FakeLanguageModel):
    async def invoke(self, request: ModelRequest) -> ModelResult:
        if self.requests:
            raise RuntimeError("summary failed")
        return await super().invoke(request)


def _document(
    document_id: str,
    *,
    name: str,
    text: str,
    content_hash: str,
    original_content: ParsedTextContent | None = None,
) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        source=ParsedSource(
            key=f"raw/{name}",
            name=name,
            file_type=Path(name).suffix.lstrip("."),
            content_sha256=content_hash,
        ),
        pages=[ParsedPage(page_index=0, text=text)],
        original_content=original_content,
    )


def test_build_wiki_pages_declares_page_only_contract():
    step = BuildWikiPages()

    assert step.name == "build_wiki_pages"
    assert {ref.key for ref in step.requirements.components} == {
        "stores.objects",
        "models.language",
    }
    assert step.requirements.artifacts == frozenset({"parsed_document_keys"})
    assert step.capabilities == StepCapabilities(
        artifacts=frozenset(
            {
                "build_wiki_pages_result",
                "wiki_page_keys",
                "wiki_index_key",
                "wiki_log_key",
            }
        )
    )


def test_build_wiki_pages_writes_page_index_and_log(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    language_model = FakeLanguageModel()
    context = FakeContext(
        {"stores.objects": object_store, "models.language": language_model}
    )
    document = _document(
        "doc_plant",
        name="plant-cell.md",
        text="# Plant Cell\n\n## Metabolism\n\nChloroplasts convert light energy.",
        content_hash="a" * 64,
    )

    async def run():
        await object_store.put("parsed/doc_plant.json", document.to_json_bytes())
        context.set_artifact("parsed_document_keys", ("parsed/doc_plant.json",))
        step = BuildWikiPages(BuildWikiPagesConfig(encoding_name="unicode"))
        await step.run(context)
        page = (await object_store.get(context.artifacts["wiki_page_keys"][0])).decode()
        index = (await object_store.get(context.artifacts["wiki_index_key"])).decode()
        log = (await object_store.get(context.artifacts["wiki_log_key"])).decode()
        return page, index, log, step.cleanup_plan(context.artifacts)

    page, index, log, cleanup = asyncio.run(run())

    result = context.artifacts["build_wiki_pages_result"]
    assert len(language_model.requests) == 1
    assert result.document_count == result.page_count == 1
    assert result.summary_model == "fake-language"
    assert context.artifacts["wiki_page_keys"] == ("wiki/pages/1-plant-cell.md",)
    assert "# Plant Cell" in page
    assert "## Summary\n\nShort biology wiki summary." in page
    assert "#### Metabolism" in page
    assert "- `raw/plant-cell.md`" in page
    assert (
        "- [Plant Cell](pages/1-plant-cell.md) — Short biology wiki summary."
        in index
    )
    assert '"page_id": "wiki_page_1"' in log
    assert '"summary": "Short biology wiki summary."' in log
    assert {target.value for target in cleanup.targets} == {
        "wiki/pages/1-plant-cell.md",
        "wiki/index.md",
        "wiki/page_log.json",
    }


def test_build_wiki_pages_resumes_completed_pages_after_failure(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    failing_model = FailingAfterFirstLanguageModel("First summary.")
    context = FakeContext(
        {"stores.objects": object_store, "models.language": failing_model}
    )
    documents = (
        _document(
            "doc_first",
            name="first.md",
            text="# First\n\nReusable body.",
            content_hash="a" * 64,
        ),
        _document(
            "doc_second",
            name="second.md",
            text="# Second\n\nNeeds a later summary.",
            content_hash="b" * 64,
        ),
    )
    step = BuildWikiPages(BuildWikiPagesConfig(summary_max_attempts=1))

    async def run_failed_then_resume():
        keys = []
        for document in documents:
            key = f"parsed/{document.document_id}.json"
            await object_store.put(key, document.to_json_bytes())
            keys.append(key)
        context.set_artifact("parsed_document_keys", tuple(keys))
        with pytest.raises(RuntimeError, match="summary failed"):
            await step.run(context)

        first_page_before = await object_store.get("wiki/pages/1-first.md")
        resume_model = FakeLanguageModel("Second summary.")
        resumed_context = FakeContext(
            {"stores.objects": object_store, "models.language": resume_model}
        )
        resumed_context.set_artifact("parsed_document_keys", tuple(keys))
        await step.run(resumed_context)
        return (
            resumed_context,
            resume_model,
            first_page_before,
            await object_store.get("wiki/pages/1-first.md"),
        )

    resumed_context, resume_model, first_page_before, first_page_after = asyncio.run(
        run_failed_then_resume()
    )

    assert first_page_after == first_page_before
    assert len(failing_model.requests) == 1
    assert len(resume_model.requests) == 1
    assert resumed_context.artifacts["wiki_page_keys"] == (
        "wiki/pages/1-first.md",
        "wiki/pages/2-second.md",
    )
    assert resumed_context.artifacts["build_wiki_pages_result"].page_count == 2


def test_build_wiki_pages_prefers_markdown_and_normalizes_headings(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    context = FakeContext({"stores.objects": object_store})
    markdown = "# Cell Biology\n\n## Organelles\n\n### Chloroplast\n\nLight reactions."
    document = _document(
        "doc_markdown",
        name="biology.pdf",
        text="flattened fallback text",
        content_hash="c" * 64,
        original_content=ParsedTextContent(
            text=markdown,
            media_type="text/markdown",
        ),
    )

    async def run():
        await object_store.put("parsed/doc_markdown.json", document.to_json_bytes())
        context.set_artifact("parsed_document_keys", ("parsed/doc_markdown.json",))
        await BuildWikiPages(
            BuildWikiPagesConfig(summary_mode="extractive", encoding_name="unicode")
        ).run(context)
        return (await object_store.get(context.artifacts["wiki_page_keys"][0])).decode()

    page = asyncio.run(run())

    assert "# Cell Biology" in page
    assert "#### Organelles" in page
    assert "##### Chloroplast" in page
    assert "flattened fallback text" not in page


def test_build_wiki_pages_can_skip_oversized_documents(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    context = FakeContext({"stores.objects": object_store})
    document = _document(
        "doc_large",
        name="large.md",
        text="123456",
        content_hash="d" * 64,
    )

    async def run(policy):
        await object_store.put("parsed/doc_large.json", document.to_json_bytes())
        context.set_artifact("parsed_document_keys", ("parsed/doc_large.json",))
        await BuildWikiPages(
            BuildWikiPagesConfig(
                summary_mode="extractive",
                max_document_tokens=5,
                encoding_name="unicode",
                oversized_document_policy=policy,
            )
        ).run(context)

    with pytest.raises(WikiDocumentLimitError, match="max_document_tokens"):
        asyncio.run(run("fail"))

    asyncio.run(run("skip"))
    result = context.artifacts["build_wiki_pages_result"]
    assert result.page_keys == ()
    assert result.skipped_document_keys == ("parsed/doc_large.json",)
    assert result.issues[0].code == "wiki_document_limit_exceeded"


def test_build_wiki_pages_config_validates_summary_mode():
    with pytest.raises(ValueError, match="summary_mode"):
        BuildWikiPagesConfig(summary_mode="bad")  # type: ignore[arg-type]
