from __future__ import annotations

import pytest

from lcfa.hybrid_cli import _parser


def test_hybrid_cli_exposes_train_rollout_and_improve() -> None:
    parser = _parser()
    help_text = parser.format_help()
    assert "typed-latent" in help_text

    train = parser.parse_args([
        "train",
        "transitions.jsonl",
        "-o",
        "controller",
        "--init-controller",
        "old-controller",
        "--backbone-mode",
        "frozen",
    ])
    assert train.command == "train"
    assert train.init_controller == "old-controller"
    assert train.backbone_mode == "frozen"
    assert train.latent_slots == 9

    rollout = parser.parse_args([
        "rollout",
        "tasks.jsonl",
        "--controller",
        "controller",
        "-o",
        "episodes",
    ])
    assert rollout.command == "rollout"
    assert rollout.max_steps == 12

    improve = parser.parse_args([
        "improve",
        "tasks.jsonl",
        "--base-transitions",
        "base.jsonl",
        "--controller",
        "controller",
        "--output-root",
        "runs",
    ])
    assert improve.command == "improve"
    assert improve.backbone_mode == "full"
    assert improve.plan_loss_weight == pytest.approx(0.25)
