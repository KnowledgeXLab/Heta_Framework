"""Compatibility exports for universal graph steps."""

from heta_framework.kb.steps.adapt_universal_graph import (
    AdaptUniversalGraph,
    AdaptUniversalGraphConfig,
    AdaptUniversalGraphResult,
    AdaptUniversalGraphForGraphRAG,
    AdaptUniversalGraphForGraphRAGConfig,
    AdaptUniversalGraphForHiRAG,
    AdaptUniversalGraphForHiRAGConfig,
    AdaptUniversalGraphForLeanRAG,
    AdaptUniversalGraphForLeanRAGConfig,
    AdaptUniversalGraphForLightRAG,
    AdaptUniversalGraphForLightRAGConfig,
)
from heta_framework.kb.steps.constrain_graph_by_ontology import (
    ConstrainGraphByOntology,
    ConstrainGraphByOntologyConfig,
    ConstrainGraphByOntologyResult,
)
from heta_framework.kb.steps.extract_universal_graph import (
    ExtractUniversalGraph,
    ExtractUniversalGraphConfig,
    ExtractUniversalGraphResult,
)
from heta_framework.kb.steps.universal_graph_common import RAGAdapterTarget

__all__ = [
    "AdaptUniversalGraph",
    "AdaptUniversalGraphConfig",
    "AdaptUniversalGraphResult",
    "AdaptUniversalGraphForGraphRAG",
    "AdaptUniversalGraphForGraphRAGConfig",
    "AdaptUniversalGraphForHiRAG",
    "AdaptUniversalGraphForHiRAGConfig",
    "AdaptUniversalGraphForLeanRAG",
    "AdaptUniversalGraphForLeanRAGConfig",
    "AdaptUniversalGraphForLightRAG",
    "AdaptUniversalGraphForLightRAGConfig",
    "ConstrainGraphByOntology",
    "ConstrainGraphByOntologyConfig",
    "ConstrainGraphByOntologyResult",
    "ExtractUniversalGraph",
    "ExtractUniversalGraphConfig",
    "ExtractUniversalGraphResult",
    "RAGAdapterTarget",
]
