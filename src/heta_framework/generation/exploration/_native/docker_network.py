"""Reserve a small, isolated subnet through Docker's atomic IPAM allocation."""

import random
import subprocess
from ipaddress import IPv4Network


def create_execution_network(pool: str, execution_id: str) -> dict:
    network = IPv4Network(pool)
    count = 1 << (28 - network.prefixlen)
    # Docker rejects overlapping allocations atomically, including other processes.
    for index in random.sample(range(count), min(count, 64)):
        subnet = str(IPv4Network((int(network.network_address) + index * 16, 28)))
        result = subprocess.run(
            [
                "docker",
                "network",
                "create",
                "--driver",
                "bridge",
                "--subnet",
                subnet,
                "--label",
                f"sci.pipeline.execution={execution_id}",
                "--label",
                f"com.docker.compose.project={execution_id}",
                execution_id,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return {"name": execution_id, "subnet": subnet}
        if "overlap" not in (result.stderr + result.stdout).lower():
            raise RuntimeError("Could not create isolated Docker network: " + result.stderr.strip())
    raise RuntimeError("No free subnet found in configured Docker network pool")
