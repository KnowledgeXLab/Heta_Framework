"""Fixed offline Docker verifier environments without host firewall requirements."""

import json

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode


class DerivationDockerEnvironment(DockerEnvironment):
    """Use Docker network_mode=none for a verifier that remains offline throughout."""

    @staticmethod
    def _requires_egress_control(*, startup_network_policy, phase_network_policies):
        policies = [startup_network_policy, *phase_network_policies]
        if all(policy.network_mode == NetworkMode.NO_NETWORK for policy in policies):
            return False
        return DockerEnvironment._requires_egress_control(
            startup_network_policy=startup_network_policy,
            phase_network_policies=phase_network_policies,
        )

    @property
    def capabilities(self):
        capabilities = super().capabilities
        policy = getattr(self, "_network_policy", None)
        if policy is not None and policy.network_mode == NetworkMode.NO_NETWORK:
            return capabilities.model_copy(
                update={
                    "disable_internet": True,
                    "dynamic_network_policy": False,
                }
            )
        return capabilities

    @property
    def _docker_compose_paths(self):
        paths = super()._docker_compose_paths
        if self.network_policy.network_mode == NetworkMode.NO_NETWORK:
            # Harbor uses a separate instance for the verifier, with a fixed policy.
            overlay = self.trial_paths.trial_dir / "heta-verifier-offline.json"
            overlay.write_text(
                json.dumps(
                    {
                        "services": {
                            "main": {
                                "network_mode": "none",
                                "labels": {
                                    "sci.pipeline.execution": self.trial_paths.trial_dir.parent.name
                                },
                            }
                        }
                    }
                )
            )
            paths = [*paths, overlay]
        return paths
