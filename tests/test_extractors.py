import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.extractors import (  # noqa: E402
    BoundingBox,
    DocumentInput,
    ExtractedAsset,
    ExtractedBlock,
    ExtractedDocument,
    render_extracted_blocks,
)


def test_extracted_document_renders_ordered_blocks():
    image = ExtractedAsset(name="chart.jpg", key="artifacts/images/chart.jpg")
    document = ExtractedDocument(
        blocks=(
            ExtractedBlock(kind="text", text="Before image"),
            ExtractedBlock(kind="image", text="A chart", asset=image),
            ExtractedBlock(kind="caption", text="Figure 1"),
        )
    )

    assert document.to_text() == (
        "Before image\n\n"
        "Image: artifacts/images/chart.jpg\n\n"
        "Image description: A chart\n\n"
        "Figure 1"
    )


def test_extractor_types_validate_inputs():
    with pytest.raises(TypeError, match="data"):
        DocumentInput(data="not-bytes", filename="doc.pdf")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="filename"):
        DocumentInput(data=b"data", filename="")
    with pytest.raises(ValueError, match="right"):
        BoundingBox(left=10, top=0, right=1, bottom=2)
    with pytest.raises(ValueError, match="name"):
        ExtractedAsset(name="")
    with pytest.raises(ValueError, match="kind"):
        ExtractedBlock(kind="")


def test_render_extracted_blocks_handles_tables():
    assert render_extracted_blocks([ExtractedBlock(kind="table", text="A | B")]) == "Table:\nA | B"
