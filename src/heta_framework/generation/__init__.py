"""Task-production contracts; concrete producers are provided separately."""

from heta_framework.generation.derivation import (
    HetaGenTaskProducerDerivation,
    HetaGenTaskProducerDerivationConfig,
)
from heta_framework.generation.exploration import (
    HetaGenTaskProducerExploration,
    HetaGenTaskProducerExplorationConfig,
)
from heta_framework.generation.graph import (
    HetaGenTaskProducerGraph,
    HetaGenTaskProducerGraphConfig,
)
from heta_framework.generation.protocols import TaskProducerProtocol
from heta_framework.generation.types import (
    InputRequirement,
    ObjectInput,
    TaskRequest,
    TaskRequirements,
    TaskResult,
)

__all__ = [
    "HetaGenTaskProducerGraph",
    "HetaGenTaskProducerGraphConfig",
    "HetaGenTaskProducerDerivation",
    "HetaGenTaskProducerDerivationConfig",
    "HetaGenTaskProducerExploration",
    "HetaGenTaskProducerExplorationConfig",
    "InputRequirement",
    "ObjectInput",
    "TaskProducerProtocol",
    "TaskRequest",
    "TaskRequirements",
    "TaskResult",
]
