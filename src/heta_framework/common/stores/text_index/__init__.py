"""Full-text index store interfaces and implementations."""

from heta_framework.common.stores.text_index.elasticsearch import (
    ElasticsearchTextIndexStore,
    ElasticsearchTextIndexStoreConfig,
)
from heta_framework.common.stores.text_index.memory import InMemoryTextIndexStore
from heta_framework.common.stores.text_index.protocols import TextIndexStoreProtocol
from heta_framework.common.stores.text_index.types import (
    TextIndexConfig,
    TextIndexRecord,
    TextQuery,
    TextSearchResult,
)

__all__ = [
    "ElasticsearchTextIndexStore",
    "ElasticsearchTextIndexStoreConfig",
    "InMemoryTextIndexStore",
    "TextIndexConfig",
    "TextIndexRecord",
    "TextIndexStoreProtocol",
    "TextQuery",
    "TextSearchResult",
]
