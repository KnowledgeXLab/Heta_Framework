"""Compatibility exports for the former HiRAG extraction step module."""

from heta_framework.kb.steps.hirag_hierarchical_aggregation import *  # noqa: F401,F403
from heta_framework.kb.steps.hirag_hierarchical_aggregation import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _parse_hirag_records,
)  # noqa: F401
