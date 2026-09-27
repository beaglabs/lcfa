from __future__ import annotations

import json

from lcfa.hybrid_train_dispatch import _preserve_init_backbone


def test_preserve_init_backbone_copies_weights_and_updates_manifest(tmp_path) -> None:
    init_dir = tmp_path / "init"
    out_dir = tmp_path / "out"
    init_dir.mkdir()
    out_dir.mkdir()

    (init_dir / "backbone.safetensors").write_bytes(b"fine-tuned-backbone")
    (init_dir / "controller.json").write_text(
        json.dumps({
            "model_id": "example/model",
            "backbone_weights": "backbone.safetensors",
        }),
        encoding="utf-8",
    )
    (out_dir / "controller.json").write_text(
        json.dumps({
            "model_id": "example/model",
            "backbone_weights": None,
            "init_backbone_loaded": True,
        }),
        encoding="utf-8",
    )

    result = _preserve_init_backbone(
        init_dir,
        out_dir,
        {
            "model_id": "example/model",
            "backbone_weights": None,
        },
    )

    assert (out_dir / "backbone.safetensors").read_bytes() == b"fine-tuned-backbone"
    assert result["backbone_weights"] == "backbone.safetensors"
    assert result["pointer_only_backbone_preserved"] is True

    manifest = json.loads((out_dir / "controller.json").read_text(encoding="utf-8"))
    assert manifest["backbone_weights"] == "backbone.safetensors"
    assert manifest["init_backbone_loaded"] is True
    assert manifest["pointer_only_backbone_preserved"] is True


def test_preserve_init_backbone_noops_without_backbone(tmp_path) -> None:
    init_dir = tmp_path / "init"
    out_dir = tmp_path / "out"
    init_dir.mkdir()
    out_dir.mkdir()
    (init_dir / "controller.json").write_text(
        json.dumps({"model_id": "example/model", "backbone_weights": None}),
        encoding="utf-8",
    )
    summary = {"model_id": "example/model", "backbone_weights": None}
    assert _preserve_init_backbone(init_dir, out_dir, summary) == summary
