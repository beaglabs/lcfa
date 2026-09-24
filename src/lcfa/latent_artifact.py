"""Artifact registration for lcfa.latent-flow.v1."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifact import ArtifactManifest, ArtifactReasoner, _resolve_weights_path, register_artifact_architecture
from .engine import LCFA
from .latent_flow import LATENT_FLOW_ARCHITECTURE, LatentFlowReasoner


def _latent_flow_factory(
    root: Path,
    manifest: ArtifactManifest,
    base_engine: LCFA | None,
    runtime_options: Mapping[str, Any],
) -> ArtifactReasoner:
    return LatentFlowReasoner.from_manifest(
        root,
        manifest,
        weights_path=_resolve_weights_path(root, manifest),
        base_engine=base_engine,
        runtime_options=runtime_options,
    )


register_artifact_architecture(LATENT_FLOW_ARCHITECTURE, _latent_flow_factory)

__all__ = ["LATENT_FLOW_ARCHITECTURE"]
