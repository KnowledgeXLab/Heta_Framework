import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (
    EmbeddingModel,
    EmbeddingModelProtocol,
    LanguageModel,
    LanguageModelProtocol,
)


def test_default_models_satisfy_model_protocols():
    language = LanguageModel(model_name="openai/test-model")
    embedding = EmbeddingModel(model_name="openai/test-embedding")

    assert isinstance(language, LanguageModelProtocol)
    assert isinstance(embedding, EmbeddingModelProtocol)
