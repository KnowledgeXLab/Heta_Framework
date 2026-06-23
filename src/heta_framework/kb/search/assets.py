"""Search assets produced by knowledge base build steps."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchAsset:
    """A queryable asset created by a build step."""

    kind: str
    name: str
    store: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind.strip() == "":
            raise ValueError("kind must not be empty")
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        if self.store is not None and self.store.strip() == "":
            raise ValueError("store must not be empty")
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "store", self.store.strip() if self.store else None)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> str:
        """Return a stable key for diagnostics."""
        return f"{self.kind}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return {
            "kind": self.kind,
            "name": self.name,
            "store": self.store,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, order=True)
class SearchAssetRef:
    """Reference to a queryable asset required by a query engine."""

    kind: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind.strip() == "":
            raise ValueError("kind must not be empty")
        if self.name is not None and self.name.strip() == "":
            raise ValueError("name must not be empty")
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "name", self.name.strip() if self.name else None)

    @property
    def key(self) -> str:
        """Return a stable key for diagnostics."""
        if self.name is None:
            return self.kind
        return f"{self.kind}:{self.name}"

    def matches(self, asset: SearchAsset) -> bool:
        """Return whether an asset satisfies this reference."""
        return asset.kind == self.kind and (self.name is None or asset.name == self.name)


class SearchAssetCollection:
    """Immutable lookup collection for queryable assets."""

    def __init__(self, assets: Iterable[SearchAsset] = ()) -> None:
        self._assets = tuple(assets)
        self._assets_by_key = {asset.key: asset for asset in self._assets}
        if len(self._assets_by_key) != len(self._assets):
            raise ValueError("search assets must not contain duplicate kind/name pairs")

    def __iter__(self) -> Iterator[SearchAsset]:
        return iter(self._assets)

    def __len__(self) -> int:
        return len(self._assets)

    @property
    def assets(self) -> tuple[SearchAsset, ...]:
        """Return all assets in stable order."""
        return self._assets

    def find(self, ref: SearchAssetRef) -> tuple[SearchAsset, ...]:
        """Return all assets that satisfy a reference."""
        return tuple(asset for asset in self._assets if ref.matches(asset))

    def contains(self, ref: SearchAssetRef) -> bool:
        """Return whether at least one asset satisfies a reference."""
        return bool(self.find(ref))

    def require(self, ref: SearchAssetRef) -> SearchAsset:
        """Return exactly one matching asset, raising if missing or ambiguous."""
        matches = self.find(ref)
        if not matches:
            raise LookupError(f"missing search asset: {ref.key}")
        if len(matches) > 1:
            names = ", ".join(asset.key for asset in matches)
            raise LookupError(f"ambiguous search asset {ref.key}; matches: {names}")
        return matches[0]

    def missing(self, refs: Iterable[SearchAssetRef]) -> tuple[SearchAssetRef, ...]:
        """Return required assets that are not satisfied by this collection."""
        return tuple(ref for ref in refs if not self.contains(ref))

    def to_list(self) -> list[dict[str, Any]]:
        """Return JSON-friendly asset dictionaries."""
        return [asset.to_dict() for asset in self._assets]
