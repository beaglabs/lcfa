"""Versioned LCFA artifacts backed by ``model.safetensors``."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from safetensors import safe_open

from .engine import LCFA
from .protocol import ExecutionContext, Finding, ReasoningPlan, SolutionState


ARTIFACT_FORMAT = "lcfa.artifact.v1"
WEIGHTS_FORMAT = "safetensors"
WEIGHTED_OUTPUT_ARCHITECTURE = "lcfa.weighted-output.v1"
STOCHASTIC_FLOW_ARCHITECTURE = "lcfa.stochastic-flow.v1"


class ArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactWeights:
    file: str
    sha256: str
    format: str = WEIGHTS_FORMAT


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    id: str
    version: str
    architecture: str
    weights: ArtifactWeights
    base_backend: str = "lcfa-zero"
    tensors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = ARTIFACT_FORMAT


class ArtifactReasoner(Protocol):
    artifact: ArtifactManifest
    metadata: Mapping[str, Any]

    def reason(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        ...


def _manifest_path(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path)
    if candidate.is_dir():
        root = candidate.resolve()
        return root, root / "artifact.json"
    manifest = candidate.resolve()
    return manifest.parent, manifest


def _weights_from_mapping(raw: Mapping[str, Any]) -> ArtifactWeights:
    file = str(raw.get("file", "model.safetensors"))
    digest = str(raw.get("sha256", ""))
    fmt = str(raw.get("format", WEIGHTS_FORMAT))
    if not digest:
        raise ArtifactError("artifact weights.sha256 is required")
    if fmt != WEIGHTS_FORMAT:
        raise ArtifactError(f"unsupported artifact weights format: {fmt!r}")
    return ArtifactWeights(file=file, sha256=digest, format=fmt)


def load_artifact_manifest(path: str | Path) -> ArtifactManifest:
    _root, manifest_path = _manifest_path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"artifact manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid artifact manifest JSON: {manifest_path}") from exc

    if not isinstance(raw, Mapping):
        raise ArtifactError("artifact manifest must be a JSON object")
    if raw.get("format") != ARTIFACT_FORMAT:
        raise ArtifactError(f"unsupported artifact format: {raw.get('format')!r}")
    weights_raw = raw.get("weights")
    if not isinstance(weights_raw, Mapping):
        raise ArtifactError("artifact weights must be an object")

    return ArtifactManifest(
        id=str(raw["id"]),
        version=str(raw["version"]),
        architecture=str(raw["architecture"]),
        weights=_weights_from_mapping(weights_raw),
        base_backend=str(raw.get("base_backend", "lcfa-zero")),
        tensors=dict(raw.get("tensors", {})),
        config=dict(raw.get("config", {})),
        metadata=dict(raw.get("metadata", {})),
    )


def _resolve_weights_path(root: Path, manifest: ArtifactManifest) -> Path:
    root = root.resolve()
    weights_path = (root / manifest.weights.file).resolve()
    if weights_path.parent != root:
        raise ArtifactError("artifact weights file must stay inside the artifact directory")
    if not weights_path.is_file():
        raise ArtifactError(f"artifact weights file not found: {weights_path}")
    actual = sha256(weights_path.read_bytes()).hexdigest()
    if actual != manifest.weights.sha256:
        raise ArtifactError(
            f"artifact weights sha256 mismatch: expected {manifest.weights.sha256}, got {actual}"
        )
    return weights_path


def _scalar_tensor(weights_path: Path, name: str) -> float:
    with safe_open(str(weights_path), framework="np", device="cpu") as handle:
        if name not in handle.keys():
            raise ArtifactError(f"artifact missing required tensor: {name}")
        tensor = handle.get_tensor(name)
        if tensor.size != 1:
            raise ArtifactError(f"artifact tensor {name} must contain exactly one scalar")
        return float(tensor.reshape(-1)[0])


def _validate_safetensors_metadata(weights_path: Path, manifest: ArtifactManifest) -> None:
    with safe_open(str(weights_path), framework="np", device="cpu") as handle:
        metadata = handle.metadata() or {}
    artifact_format = metadata.get("format")
    architecture = metadata.get("architecture")
    artifact_id = metadata.get("id")
    if artifact_format is not None and artifact_format != ARTIFACT_FORMAT:
        raise ArtifactError(f"safetensors metadata format mismatch: {artifact_format!r}")
    if architecture is not None and architecture != manifest.architecture:
        raise ArtifactError(
            f"safetensors architecture mismatch: manifest={manifest.architecture!r}, weights={architecture!r}"
        )
    if artifact_id is not None and artifact_id != manifest.id:
        raise ArtifactError(
            f"safetensors artifact id mismatch: manifest={manifest.id!r}, weights={artifact_id!r}"
        )


def _affine_float(value: Any, scale: float, bias: float) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return value * scale + bias
    if isinstance(value, list):
        return [_affine_float(item, scale, bias) for item in value]
    if isinstance(value, tuple):
        return tuple(_affine_float(item, scale, bias) for item in value)
    if isinstance(value, Mapping):
        return {key: _affine_float(item, scale, bias) for key, item in value.items()}
    return value


class SafetensorsReasonerAdapter:
    """Reference weight-backed reasoner satisfying the standard LCFA contract."""

    def __init__(
        self,
        artifact: ArtifactManifest,
        *,
        weights_path: Path,
        base_engine: LCFA | None = None,
    ) -> None:
        if artifact.architecture != WEIGHTED_OUTPUT_ARCHITECTURE:
            raise ArtifactError(f"unsupported built-in architecture: {artifact.architecture!r}")
        if artifact.base_backend != "lcfa-zero":
            raise ArtifactError(f"unsupported base backend: {artifact.base_backend!r}")
        self.artifact = artifact
        self.weights_path = weights_path
        self.base_engine = base_engine or LCFA()
        _validate_safetensors_metadata(weights_path, artifact)
        self.numeric_scale = _scalar_tensor(weights_path, "numeric.scale")
        self.numeric_bias = _scalar_tensor(weights_path, "numeric.bias")
        self.metadata = {
            "backend": "safetensors",
            "artifact_id": artifact.id,
            "artifact_version": artifact.version,
            "architecture": artifact.architecture,
            "weights_sha256": artifact.weights.sha256,
        }

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        base_engine: LCFA | None = None,
    ) -> "SafetensorsReasonerAdapter":
        root, _manifest_path_value = _manifest_path(path)
        manifest = load_artifact_manifest(path)
        weights_path = _resolve_weights_path(root, manifest)
        return cls(manifest, weights_path=weights_path, base_engine=base_engine)

    def reason(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        solution = self.base_engine.reason(plan, context)
        values = {
            key: _affine_float(value, self.numeric_scale, self.numeric_bias)
            for key, value in solution.values.items()
        }
        findings = tuple(
            replace(
                finding,
                value=_affine_float(finding.value, self.numeric_scale, self.numeric_bias),
                confidence=(
                    _affine_float(finding.confidence, self.numeric_scale, self.numeric_bias)
                    if finding.confidence is not None
                    else None
                ),
            )
            if isinstance(finding, Finding)
            else finding
            for finding in solution.findings
        )
        return replace(
            solution,
            values=values,
            findings=findings,
            metadata={**solution.metadata, "artifact": dict(self.metadata)},
        )


ArtifactFactory = Callable[
    [Path, ArtifactManifest, LCFA | None, Mapping[str, Any]], ArtifactReasoner
]
_ARCHITECTURES: dict[str, ArtifactFactory] = {}


def register_artifact_architecture(name: str, factory: ArtifactFactory) -> None:
    if not name:
        raise ValueError("artifact architecture name must not be empty")
    _ARCHITECTURES[name] = factory


def artifact_architectures() -> tuple[str, ...]:
    return tuple(sorted(_ARCHITECTURES))


def _weighted_factory(
    root: Path,
    manifest: ArtifactManifest,
    base_engine: LCFA | None,
    _runtime_options: Mapping[str, Any],
) -> ArtifactReasoner:
    return SafetensorsReasonerAdapter(
        manifest,
        weights_path=_resolve_weights_path(root, manifest),
        base_engine=base_engine,
    )


def _stochastic_factory(
    root: Path,
    manifest: ArtifactManifest,
    base_engine: LCFA | None,
    runtime_options: Mapping[str, Any],
) -> ArtifactReasoner:
    from .stochastic import StochasticFlowReasoner

    return StochasticFlowReasoner.from_manifest(
        root,
        manifest,
        weights_path=_resolve_weights_path(root, manifest),
        base_engine=base_engine,
        runtime_options=runtime_options,
    )


def load_artifact_reasoner(
    path: str | Path,
    *,
    base_engine: LCFA | None = None,
    runtime_options: Mapping[str, Any] | None = None,
) -> ArtifactReasoner:
    root, _manifest_path_value = _manifest_path(path)
    manifest = load_artifact_manifest(path)
    try:
        factory = _ARCHITECTURES[manifest.architecture]
    except KeyError as exc:
        available = ", ".join(sorted(_ARCHITECTURES)) or "none"
        raise ArtifactError(
            f"unsupported artifact architecture {manifest.architecture!r}; registered: {available}"
        ) from exc
    return factory(root, manifest, base_engine, dict(runtime_options or {}))


register_artifact_architecture(WEIGHTED_OUTPUT_ARCHITECTURE, _weighted_factory)
register_artifact_architecture(STOCHASTIC_FLOW_ARCHITECTURE, _stochastic_factory)


__all__ = [
    "ARTIFACT_FORMAT",
    "STOCHASTIC_FLOW_ARCHITECTURE",
    "WEIGHTED_OUTPUT_ARCHITECTURE",
    "ArtifactError",
    "ArtifactManifest",
    "ArtifactReasoner",
    "ArtifactWeights",
    "SafetensorsReasonerAdapter",
    "artifact_architectures",
    "load_artifact_manifest",
    "load_artifact_reasoner",
    "register_artifact_architecture",
]
