"""Transition datasets for training recurrent LCFA controllers.

The semantic agent records ``lcfa.semantic-trajectory.v1`` episodes.  This
module turns those episodes into one-step supervision records suitable for a
recurrent controller: previous event/state -> next action + stop/value targets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RECURRENT_TRANSITION_FORMAT = "lcfa.recurrent-transition.v1"

ACTION_VOCAB: tuple[str, ...] = (
    "repo.read",
    "repo.search",
    "repo.replace",
    "repo.edit",
    "test.run",
    "git.status",
    "git.diff",
    "process.exec",
    "docs.fetch",
    "stop",
)


@dataclass(frozen=True, slots=True)
class RecurrentTransition:
    episode_id: str
    step_index: int
    goal: str
    event: Mapping[str, Any]
    target_action: str
    stop_target: bool
    value_target: float | None = None
    metadata: Mapping[str, Any] | None = None
    schema_version: str = RECURRENT_TRANSITION_FORMAT

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _episode_success(episode: Mapping[str, Any]) -> float | None:
    """Read an optional externally supplied benchmark outcome.

    Semantic episodes intentionally do not infer success from ``final`` or from
    producing a non-empty patch.  A value target is valid only when a grader or
    caller explicitly attaches one as ``success``, ``resolved``, or
    ``metadata.success``.
    """
    for key in ("success", "resolved"):
        if key in episode:
            value = episode[key]
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
    metadata = _mapping(episode.get("metadata"))
    value = metadata.get("success")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return None


def episode_to_transitions(episode: Mapping[str, Any]) -> tuple[RecurrentTransition, ...]:
    if str(episode.get("schema_version", "")) != "lcfa.semantic-trajectory.v1":
        raise ValueError("expected lcfa.semantic-trajectory.v1 episode")
    episode_id = str(episode.get("id", ""))
    goal = str(episode.get("goal", ""))
    if not episode_id or not goal:
        raise ValueError("semantic episode requires id and goal")
    steps_raw = episode.get("steps", ())
    if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, (str, bytes)):
        raise ValueError("semantic episode steps must be an array")

    value_target = _episode_success(episode)
    previous: Mapping[str, Any] = {
        "kind": "goal",
        "goal": goal,
    }
    out: list[RecurrentTransition] = []
    for position, raw in enumerate(steps_raw, start=1):
        step = _mapping(raw)
        action = _mapping(step.get("action"))
        action_name = str(action.get("name") or "stop")
        if action_name not in ACTION_VOCAB:
            # Preserve a closed action vocabulary so a controller head has a
            # stable output dimension. Unknown actions are not silently folded.
            raise ValueError(f"unsupported recurrent target action: {action_name}")
        terminal = bool(step.get("terminal", False))
        event = {
            "previous": previous,
            "solution_id": step.get("solution_id"),
            "hypothesis": step.get("hypothesis"),
        }
        out.append(
            RecurrentTransition(
                episode_id=episode_id,
                step_index=int(step.get("index", position)),
                goal=goal,
                event=event,
                target_action=action_name,
                stop_target=terminal,
                value_target=value_target,
                metadata={"has_observation": bool(step.get("observation"))},
            )
        )
        previous = {
            "kind": "transition",
            "action": dict(action) if action else None,
            "hypothesis": step.get("hypothesis"),
            "observation": step.get("observation") or {},
            "terminal": terminal,
        }
    return tuple(out)


def load_episode(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"episode must be a JSON object: {path}")
    return value


def load_transitions(path: str | Path) -> tuple[RecurrentTransition, ...]:
    rows: list[RecurrentTransition] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, Mapping):
                raise ValueError(f"transition line {line_number} is not an object")
            if str(raw.get("schema_version", "")) != RECURRENT_TRANSITION_FORMAT:
                raise ValueError(f"transition line {line_number} has wrong schema_version")
            rows.append(
                RecurrentTransition(
                    episode_id=str(raw["episode_id"]),
                    step_index=int(raw["step_index"]),
                    goal=str(raw["goal"]),
                    event=dict(_mapping(raw.get("event"))),
                    target_action=str(raw["target_action"]),
                    stop_target=bool(raw["stop_target"]),
                    value_target=(None if raw.get("value_target") is None else float(raw["value_target"])),
                    metadata=dict(_mapping(raw.get("metadata"))),
                )
            )
    return tuple(rows)


def dump_transitions(rows: Iterable[RecurrentTransition], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_transition_file(episodes: Sequence[str | Path], output: str | Path) -> int:
    rows: list[RecurrentTransition] = []
    for path in episodes:
        rows.extend(episode_to_transitions(load_episode(path)))
    return dump_transitions(rows, output)


__all__ = [
    "ACTION_VOCAB",
    "RECURRENT_TRANSITION_FORMAT",
    "RecurrentTransition",
    "dump_transitions",
    "episode_to_transitions",
    "load_episode",
    "load_transitions",
    "prepare_transition_file",
]
