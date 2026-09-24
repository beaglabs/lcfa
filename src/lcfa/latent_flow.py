"""Latent-first LCFA runtime built around modality-neutral zplugs."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from blake3 import blake3
from safetensors import safe_open

from .artifact import ArtifactError, ArtifactManifest
from .engine import LCFA
from .protocol import ExecutionContext, ReasoningPlan, SolutionState
from .zplug import HashTextZPlug, LatentPacket, MLXTextZPlug, Observation, ZContext, ZPlug

LATENT_FLOW_ARCHITECTURE = "lcfa.latent-flow.v1"
LATENT_STATE_FORMAT = "lcfa.latent-state.v1"


@dataclass(frozen=True, slots=True)
class LatentState:
    id: str
    shared: tuple[float, ...]
    packets: tuple[LatentPacket, ...] = ()
    step: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = LATENT_STATE_FORMAT

    @property
    def dim(self) -> int:
        return len(self.shared)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _load_tensor(handle: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    if name not in handle.keys():
        raise ArtifactError(f"latent-flow artifact missing tensor: {name}")
    value = np.asarray(handle.get_tensor(name), dtype=np.float32)
    if ndim is not None and value.ndim != ndim:
        raise ArtifactError(f"latent-flow tensor {name!r} must have ndim={ndim}, got {value.ndim}")
    return value


class NumpyLatentCore:
    """Portable inference core for text-first latent-flow checkpoints.

    Training can happen in MLX; inference here deliberately uses NumPy so the
    learned latent checkpoint remains backend-independent and testable in CI.
    """

    def __init__(self, weights_path: str | Path) -> None:
        with safe_open(str(weights_path), framework="np", device="cpu") as handle:
            self.encoder_weight = _load_tensor(handle, "encoder.weight", ndim=2)
            self.encoder_bias = _load_tensor(handle, "encoder.bias", ndim=1)
            self.dyn0_weight = _load_tensor(handle, "dynamics.0.weight", ndim=2)
            self.dyn0_bias = _load_tensor(handle, "dynamics.0.bias", ndim=1)
            self.dyn1_weight = _load_tensor(handle, "dynamics.1.weight", ndim=2)
            self.dyn1_bias = _load_tensor(handle, "dynamics.1.bias", ndim=1)
            metadata = handle.metadata() or {}
        latent_dim, feature_dim = self.encoder_weight.shape
        if self.encoder_bias.shape != (latent_dim,):
            raise ArtifactError("encoder.bias shape does not match encoder.weight")
        if self.dyn0_weight.shape[1] != latent_dim:
            raise ArtifactError("dynamics.0 input dimension must equal latent dimension")
        hidden_dim = self.dyn0_weight.shape[0]
        if self.dyn0_bias.shape != (hidden_dim,):
            raise ArtifactError("dynamics.0.bias shape mismatch")
        if self.dyn1_weight.shape != (latent_dim, hidden_dim):
            raise ArtifactError("dynamics.1.weight shape mismatch")
        if self.dyn1_bias.shape != (latent_dim,):
            raise ArtifactError("dynamics.1.bias shape mismatch")
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.metadata = dict(metadata)

    def encode(self, features: tuple[float, ...] | np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.shape != (self.feature_dim,):
            raise ArtifactError(
                f"zplug feature dimension mismatch: artifact expects {self.feature_dim}, got {x.shape}"
            )
        z = self.encoder_weight @ x + self.encoder_bias
        norm = float(np.linalg.norm(z))
        return (z / norm if norm > 1e-8 else z).astype(np.float32)

    def step(self, latent: np.ndarray) -> np.ndarray:
        z = np.asarray(latent, dtype=np.float32)
        hidden = _gelu(self.dyn0_weight @ z + self.dyn0_bias)
        predicted = self.dyn1_weight @ hidden + self.dyn1_bias
        norm = float(np.linalg.norm(predicted))
        return (predicted / norm if norm > 1e-8 else predicted).astype(np.float32)

    def evolve(self, features: tuple[float, ...] | np.ndarray, *, steps: int) -> np.ndarray:
        z = self.encode(features)
        for _ in range(max(0, int(steps))):
            z = self.step(z)
        return z


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _safe(v) for k, v in vars(value).items()}
    return repr(value)


def _latent_id(values: np.ndarray) -> str:
    digest = blake3(np.asarray(values, dtype=np.float32).tobytes()).hexdigest(length=16)
    return f"latent:{digest}"


class LatentFlowReasoner:
    """Text-first latent-flow runtime preserving the typed LCFA solution contract.

    The initial implementation learns/executes the latent representation and
    dynamics but deliberately does not let an untrained latent decoder overwrite
    grounded LCFA values. A future solution decoder can be promoted behind the
    same architecture once it passes benchmark gates.
    """

    def __init__(self, artifact: ArtifactManifest, *, weights_path: Path,
                 zplug: ZPlug, base_engine: LCFA | None = None,
                 steps: int = 4, include_vector: bool = False) -> None:
        if artifact.architecture != LATENT_FLOW_ARCHITECTURE:
            raise ArtifactError(f"wrong latent-flow architecture: {artifact.architecture!r}")
        if artifact.base_backend != "lcfa-zero":
            raise ArtifactError(f"unsupported latent-flow base backend: {artifact.base_backend!r}")
        self.artifact = artifact
        self.weights_path = weights_path
        self.base_engine = base_engine or LCFA()
        self.zplug = zplug
        self.core = NumpyLatentCore(weights_path)
        self.steps = max(0, int(steps))
        self.include_vector = bool(include_vector)
        self.metadata = {
            "backend": "latent-flow",
            "architecture": artifact.architecture,
            "artifact_id": artifact.id,
            "artifact_version": artifact.version,
            "weights_sha256": artifact.weights.sha256,
            "zplug": {
                "id": zplug.manifest.id,
                "version": zplug.manifest.version,
                "modalities": list(zplug.manifest.modalities),
                "capabilities": list(zplug.manifest.capabilities),
                "metadata": dict(zplug.manifest.metadata),
            },
            "feature_dim": self.core.feature_dim,
            "latent_dim": self.core.latent_dim,
            "latent_steps": self.steps,
        }

    @classmethod
    def from_manifest(cls, root: Path, manifest: ArtifactManifest, *, weights_path: Path,
                      base_engine: LCFA | None = None,
                      runtime_options: Mapping[str, Any] | None = None) -> "LatentFlowReasoner":
        runtime = dict(runtime_options or {})
        config = dict(manifest.config)
        zcfg = dict(config.get("zplug", {}))
        flow = dict(config.get("flow", {}))
        ztype = str(runtime.get("zplug_type") or zcfg.get("type", "hash-text"))
        if ztype == "hash-text":
            zplug: ZPlug = HashTextZPlug(dim=int(zcfg.get("feature_dim", 64)))
        elif ztype == "mlx-text":
            raw_path = runtime.get("zplug_model_path") or runtime.get("backbone_path") or zcfg.get("model_path") or zcfg.get("path")
            if not raw_path:
                raise ArtifactError("mlx-text zplug requires zplug.model_path or --zplug-model-path")
            candidate = Path(str(raw_path))
            model_path = candidate if candidate.is_absolute() else (root / candidate).resolve()
            zplug = MLXTextZPlug(
                model_path,
                max_tokens=int(runtime.get("zplug_max_tokens") or zcfg.get("max_tokens", 2048)),
                pool=str(zcfg.get("pool", "last")),
                normalize=bool(zcfg.get("normalize", True)),
            )
        else:
            raise ArtifactError(f"unsupported latent-flow zplug type: {ztype!r}")
        return cls(
            manifest,
            weights_path=weights_path,
            zplug=zplug,
            base_engine=base_engine,
            steps=int(runtime.get("latent_steps") or flow.get("steps", 4)),
            include_vector=bool(runtime.get("include_latent_vector", flow.get("include_vector", False))),
        )

    def _text_observation(self, plan: ReasoningPlan, context: ExecutionContext,
                          anchor: SolutionState) -> Observation:
        query = context.metadata.get("query") or plan.metadata.get("query") or plan.metadata.get("task") or plan.id
        payload = json.dumps({
            "query": query,
            "plan_id": plan.id,
            "plan_metadata": _safe(plan.metadata),
            "state": _safe(context.state),
            "anchor_values": _safe(anchor.values),
            "evidence_ids": [item.id for item in anchor.evidence],
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return Observation(
            id=f"{plan.id}:text",
            modality="text",
            payload=payload,
            media_type="application/json",
            evidence=anchor.evidence,
            metadata={"query": str(query)},
        )

    def reason(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        anchor = self.base_engine.reason(plan, context)
        observation = self._text_observation(plan, context, anchor)
        packet = self.zplug.encode(observation, ZContext(query=observation.metadata.get("query")))
        latent = self.core.evolve(packet.features, steps=self.steps)
        state = LatentState(
            id=_latent_id(latent),
            shared=tuple(float(v) for v in latent),
            packets=(packet,),
            step=self.steps,
            metadata={"zplug_id": packet.zplug_id},
        )
        flow_metadata: dict[str, Any] = {
            "status": "representation-ready",
            "architecture": LATENT_FLOW_ARCHITECTURE,
            "state_id": state.id,
            "step": state.step,
            "feature_dim": packet.feature_dim,
            "latent_dim": state.dim,
            "latent_norm": float(np.linalg.norm(latent)),
            "zplug_id": packet.zplug_id,
            "training_ready": True,
            "solution_decoder_active": False,
        }
        if self.include_vector:
            flow_metadata["shared"] = list(state.shared)
        return replace(anchor, metadata={
            **anchor.metadata,
            "artifact": dict(self.metadata),
            "latent_flow": flow_metadata,
        })


__all__ = [
    "LATENT_FLOW_ARCHITECTURE", "LATENT_STATE_FORMAT", "LatentFlowReasoner", "LatentState",
    "NumpyLatentCore",
]
