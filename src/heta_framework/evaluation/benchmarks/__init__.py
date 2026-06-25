"""Built-in benchmark adapters."""

from heta_framework.evaluation.benchmarks.beir import (
    BEIR_DATASET_BASE_URL,
    BEIR_GITHUB_URL,
    BEIR_RECOMMENDED_DATASETS,
    BeirBenchmark,
)
from heta_framework.evaluation.benchmarks.jsonl import JsonlBenchmark
from heta_framework.evaluation.benchmarks.multihop_rag import (
    MULTIHOP_RAG_CORPUS_URL,
    MULTIHOP_RAG_QUERIES_URL,
    MultiHopRagBenchmark,
)
from heta_framework.evaluation.benchmarks.uda import (
    UDA_EXTENDED_BENCH_BASE_URL,
    UDA_GITHUB_URL,
    UDA_QA_BASE_URL,
    UDA_SOURCE_DOCS_URL,
    UDA_SOURCE_DOCS_REPO_ID,
    UdaBenchmark,
    UdaSubset,
)

__all__ = [
    "BEIR_DATASET_BASE_URL",
    "BEIR_GITHUB_URL",
    "BEIR_RECOMMENDED_DATASETS",
    "BeirBenchmark",
    "JsonlBenchmark",
    "MULTIHOP_RAG_CORPUS_URL",
    "MULTIHOP_RAG_QUERIES_URL",
    "MultiHopRagBenchmark",
    "UDA_EXTENDED_BENCH_BASE_URL",
    "UDA_GITHUB_URL",
    "UDA_QA_BASE_URL",
    "UDA_SOURCE_DOCS_URL",
    "UDA_SOURCE_DOCS_REPO_ID",
    "UdaBenchmark",
    "UdaSubset",
]
