"""LCFA latent-space extension (zplug) contracts and built-in text zplugs."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from blake3 import blake3

from .protocol import EntityRef, EvidenceRef

ZPLUG_FORMAT = "lcfa.zplug.v1"
LATENT_PACKET_FORMAT = "lcfa.latent-packet.v1"


class ZPlugError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    modality: str
    payload: Any
    media_type: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    timestamp: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ZContext:
    query: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LatentPacket:
    id: str
    zplug_id: str
    modality: str
    features: tuple[float, ...]
    shared: tuple[float, ...] = ()
    private: tuple[float, ...] = ()
    entity_refs: tuple[EntityRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    timestamp: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = LATENT_PACKET_FORMAT

    @property
    def feature_dim(self) -> int:
        return len(self.features)


@dataclass(frozen=True, slots=True)
class LatentDelta:
    shared_delta: tuple[float, ...] = ()
    private_delta: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutputRequest:
    kind: str
    modality: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ZPlugManifest:
    id: str
    version: str
    modalities: tuple[str, ...]
    capabilities: tuple[str, ...]
    priority: int = 0
    requires_network: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = ZPLUG_FORMAT

    def supports(self, capability: str, modality: str | None = None) -> bool:
        if capability not in self.capabilities:
            return False
        return modality is None or modality in self.modalities or "*" in self.modalities


class ZPlug(Protocol):
    manifest: ZPlugManifest

    def encode(self, observation: Observation, context: ZContext) -> LatentPacket: ...


class ZPlugRegistry:
    def __init__(self) -> None:
        self._plugs: dict[str, ZPlug] = {}

    def register(self, plug: ZPlug) -> None:
        plug_id = plug.manifest.id
        if not plug_id:
            raise ZPlugError("zplug id must not be empty")
        if plug_id in self._plugs:
            raise ZPlugError(f"zplug already registered: {plug_id}")
        self._plugs[plug_id] = plug

    def get(self, plug_id: str) -> ZPlug:
        try:
            return self._plugs[plug_id]
        except KeyError as exc:
            raise ZPlugError(f"unknown zplug: {plug_id}") from exc

    def resolve(self, modality: str, *, capability: str | None = None) -> ZPlug:
        capability = capability or f"encode:{modality}"
        matches = [
            plug for plug in self._plugs.values()
            if plug.manifest.supports(capability, modality)
        ]
        if not matches:
            raise ZPlugError(f"no zplug for modality={modality!r} capability={capability!r}")
        matches.sort(key=lambda p: (-p.manifest.priority, p.manifest.id))
        return matches[0]

    def encode(self, observation: Observation, *, context: ZContext | None = None,
               zplug_id: str | None = None) -> LatentPacket:
        plug = self.get(zplug_id) if zplug_id else self.resolve(observation.modality)
        return plug.encode(observation, context or ZContext())

    def manifests(self) -> tuple[ZPlugManifest, ...]:
        return tuple(self._plugs[key].manifest for key in sorted(self._plugs))


class HashTextZPlug:
    """Deterministic text zplug used for CI and contract tests, not quality inference."""

    def __init__(self, *, dim: int = 64, plug_id: str = "lcfa.text.hash") -> None:
        if dim < 8:
            raise ValueError("hash text zplug dim must be >= 8")
        self.dim = int(dim)
        self.manifest = ZPlugManifest(
            id=plug_id,
            version="1.0.0",
            modalities=("text",),
            capabilities=("encode:text", "enrich:text"),
            metadata={"backend": "hash", "quality": "reference-only", "feature_dim": self.dim},
        )

    def encode(self, observation: Observation, context: ZContext) -> LatentPacket:
        if observation.modality != "text":
            raise ZPlugError("HashTextZPlug accepts only text observations")
        text = str(observation.payload)
        vec = np.zeros((self.dim,), dtype=np.float32)
        tokens = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
        if not tokens:
            tokens = [""]
        for token in tokens:
            digest = blake3(token.encode("utf-8")).digest(length=8)
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return LatentPacket(
            id=f"{observation.id}:latent",
            zplug_id=self.manifest.id,
            modality="text",
            features=tuple(float(v) for v in vec),
            evidence_refs=observation.evidence,
            timestamp=observation.timestamp,
            metadata={"token_count": len(tokens), "context_query": context.query},
        )


class MLXTextZPlug:
    """Frozen local MLX-LM hidden-state text enricher.

    The implementation intentionally reads the final hidden state before the LM
    head (`model.model(tokens)`) and pools it. This keeps generation out of the
    latent path: the backbone acts as a semantic feature source, not the reasoner.
    """

    def __init__(self, model_path: str | Path, *, max_tokens: int = 2048,
                 pool: str = "last", normalize: bool = True,
                 plug_id: str = "lcfa.text.mlx-hidden") -> None:
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError as exc:
            raise ZPlugError("MLX text zplug requires `pip install -e '.[mlx]'`") from exc
        self._mx = mx
        self.model_path = str(model_path)
        self.model, self.tokenizer = load(self.model_path)
        if not hasattr(self.model, "model"):
            raise ZPlugError("selected MLX-LM model does not expose a pre-LM-head `.model` encoder")
        self.max_tokens = max(32, int(max_tokens))
        if pool not in {"last", "mean"}:
            raise ValueError("MLX text zplug pool must be 'last' or 'mean'")
        self.pool = pool
        self.normalize = bool(normalize)
        hidden_size = int(getattr(getattr(self.model, "args", None), "hidden_size", 0) or 0)
        self.manifest = ZPlugManifest(
            id=plug_id,
            version="1.0.0",
            modalities=("text",),
            capabilities=("encode:text", "enrich:text"),
            metadata={
                "backend": "mlx-lm-hidden",
                "model_path": self.model_path,
                "pool": self.pool,
                "max_tokens": self.max_tokens,
                "feature_dim": hidden_size or None,
                "frozen": True,
            },
        )

    def _tokens(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)
        ids = list(int(v) for v in encoded)
        return ids[-self.max_tokens:] or [0]

    def encode(self, observation: Observation, context: ZContext) -> LatentPacket:
        if observation.modality != "text":
            raise ZPlugError("MLXTextZPlug accepts only text observations")
        ids = self._tokens(str(observation.payload))
        tokens = self._mx.array([ids])
        hidden = self.model.model(tokens)
        vector = hidden[0, -1, :] if self.pool == "last" else self._mx.mean(hidden[0], axis=0)
        self._mx.eval(vector)
        arr = np.array(vector, dtype=np.float32)
        if self.normalize:
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr /= norm
        return LatentPacket(
            id=f"{observation.id}:latent",
            zplug_id=self.manifest.id,
            modality="text",
            features=tuple(float(v) for v in arr),
            evidence_refs=observation.evidence,
            timestamp=observation.timestamp,
            metadata={
                "token_count": len(ids),
                "pool": self.pool,
                "model_path": self.model_path,
                "context_query": context.query,
            },
        )


def zplug_manifest_to_dict(manifest: ZPlugManifest) -> dict[str, Any]:
    return {
        "format": manifest.format,
        "id": manifest.id,
        "version": manifest.version,
        "modalities": list(manifest.modalities),
        "capabilities": list(manifest.capabilities),
        "priority": manifest.priority,
        "requires_network": manifest.requires_network,
        "metadata": dict(manifest.metadata),
    }


def zplug_manifest_from_dict(raw: Mapping[str, Any]) -> ZPlugManifest:
    if raw.get("format") != ZPLUG_FORMAT:
        raise ZPlugError(f"unsupported zplug manifest format: {raw.get('format')!r}")
    return ZPlugManifest(
        id=str(raw["id"]),
        version=str(raw["version"]),
        modalities=tuple(str(v) for v in raw.get("modalities", ())),
        capabilities=tuple(str(v) for v in raw.get("capabilities", ())),
        priority=int(raw.get("priority", 0)),
        requires_network=bool(raw.get("requires_network", False)),
        metadata=dict(raw.get("metadata", {})),
    )


def load_zplug_manifest(path: str | Path) -> ZPlugManifest:
    return zplug_manifest_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "LATENT_PACKET_FORMAT", "ZPLUG_FORMAT", "HashTextZPlug", "LatentDelta", "LatentPacket",
    "MLXTextZPlug", "Observation", "OutputRequest", "ZContext", "ZPlug", "ZPlugError",
    "ZPlugManifest", "ZPlugRegistry", "load_zplug_manifest", "zplug_manifest_from_dict",
    "zplug_manifest_to_dict",
]
