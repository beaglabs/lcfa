"""Transition datasets for training recurrent LCFA controllers.

The semantic agent records ``lcfa.semantic-trajectory.v1`` episodes. This
module turns those episodes into one-step supervision records suitable for a
recurrent controller: event_t -> action_t / stop_t / value_t.

The controller event stream intentionally excludes teacher/oracle hypotheses
and semantic cognition snapshots. Action supervision may only depend on the
goal plus prior actions/observations that are also available during rollout.
Large write payloads and observations are compacted before they are fed back
through RWKV so recurrent state does not repeatedly tokenize whole files,
diffs, or process logs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RECURRENT_TRANSITION_FORMAT = "lcfa.recurrent-transition.v2"
LEGACY_RECURRENT_TRANSITION_FORMAT = "lcfa.recurrent-transition.v1"
MAX_RECURRENT_TEXT_CHARS = 4096
MAX_RECURRENT_SEQUENCE_ITEMS = 24

ACTION_VOCAB: tuple[str, ...] = (
    "repo.read",
    "repo.search",
    "repo.replace",
    "repo.edit",
    "test.run",
    "verify.run",
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


def _text_summary(value: str) -> Mapping[str, Any]:
    raw = value.encode("utf-8")
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _compact_text(value: str) -> str | Mapping[str, Any]:
    if len(value) <= MAX_RECURRENT_TEXT_CHARS:
        return value
    raw = value.encode("utf-8")
    half = MAX_RECURRENT_TEXT_CHARS // 2
    preview = value[:half] + "\n...<truncated>...\n" + value[-half:]
    return {
        "preview": preview,
        "chars": len(value),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "truncated": True,
    }


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    """Bound arbitrary action observations while preserving useful evidence."""
    if isinstance(value, str):
        return _compact_text(value)
    if isinstance(value, Mapping):
        if depth >= 8:
            rendered = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
            return _compact_text(rendered)
        return {
            str(key): _compact_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        compacted = [
            _compact_value(item, depth=depth + 1)
            for item in items[:MAX_RECURRENT_SEQUENCE_ITEMS]
        ]
        if len(items) <= MAX_RECURRENT_SEQUENCE_ITEMS:
            return compacted
        return {
            "items": compacted,
            "total_items": len(items),
            "truncated": True,
        }
    return value


def compact_action(action: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Return an action representation safe to feed back into recurrent state.

    Read/search/test actions retain their inputs. Write actions keep their
    semantic target and fingerprints of the written text, but not the complete
    source body.
    """
    if not isinstance(action, Mapping):
        return None
    name = str(action.get("name") or "")
    inputs = dict(_mapping(action.get("inputs")))
    if name == "repo.edit" and isinstance(inputs.get("content"), str):
        summary = _text_summary(inputs.pop("content"))
        inputs["content_bytes"] = summary["bytes"]
        inputs["content_sha256"] = summary["sha256"]
    elif name == "repo.replace":
        for key in ("old", "new"):
            value = inputs.pop(key, None)
            if isinstance(value, str):
                summary = _text_summary(value)
                inputs[f"{key}_bytes"] = summary["bytes"]
                inputs[f"{key}_sha256"] = summary["sha256"]
            elif value is not None:
                inputs[key] = value
    return {"name": name, "inputs": _compact_value(inputs)}


def _named_observation(
    observation: Mapping[str, Any],
    key: str,
) -> Any | None:
    nested = observation.get("observations")
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    if key in observation:
        return observation[key]
    results = observation.get("results")
    if isinstance(results, Mapping) and len(results) == 1:
        return next(iter(results.values()))
    return None


def compact_observation(
    action: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Deduplicate executor envelopes and bound recurrent observation size.

    Oracle/executor observations can contain the same payload in both
    ``results`` and ``observations``. The recurrent controller only needs one
    semantic copy. Workspace actions themselves still retain their full output;
    this compaction affects only the RWKV event stream.
    """
    if not isinstance(observation, Mapping):
        return {}
    action_name = str(action.get("name") or "") if isinstance(action, Mapping) else ""
    key_by_action = {
        "repo.read": "file",
        "repo.search": "search",
        "repo.edit": "edit",
        "repo.replace": "edit",
        "process.exec": "process",
        "test.run": "process",
        "verify.run": "process",
    }
    key = key_by_action.get(action_name)
    if key:
        payload = _named_observation(observation, key)
        if payload is not None:
            return {key: _compact_value(payload)}
    return dict(_compact_value(observation))


def normalize_event(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the rollout-available, bounded recurrent event schema.

    Legacy datasets may contain ``hypothesis`` or ``cognition`` fields. They
    are stripped at load time so old files cannot accidentally reintroduce
    teacher information into training. Training and live rollout call this same
    function, so their recurrent input schema remains identical.
    """
    kind = str(event.get("kind") or "")
    if kind == "goal":
        return {"kind": "goal", "goal": _compact_text(str(event.get("goal") or ""))}
    if kind == "transition":
        raw_action = event.get("action") if isinstance(event.get("action"), Mapping) else None
        observation = event.get("observation")
        return {
            "kind": "transition",
            "action": compact_action(raw_action),
            "observation": compact_observation(
                raw_action,
                observation if isinstance(observation, Mapping) else None,
            ),
        }
    compacted = _compact_value(event)
    return dict(compacted) if isinstance(compacted, Mapping) else {"value": compacted}


def _episode_success(episode: Mapping[str, Any]) -> float | None:
    """Read an optional externally supplied benchmark outcome.

    We never infer success from ``final`` or from producing a patch. A value
    target is valid only when a grader/caller explicitly attaches one.
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
    previous_action: Mapping[str, Any] | None = None
    previous_observation: Mapping[str, Any] = {}
    out: list[RecurrentTransition] = []
    for position, raw in enumerate(steps_raw, start=1):
        step = _mapping(raw)
        action = _mapping(step.get("action"))
        action_name = str(action.get("name") or "stop")
        if action_name not in ACTION_VOCAB:
            raise ValueError(f"unsupported recurrent target action: {action_name}")
        terminal = bool(step.get("terminal", False))

        if position == 1:
            event: Mapping[str, Any] = normalize_event({"kind": "goal", "goal": goal})
        else:
            event = normalize_event({
                "kind": "transition",
                "action": dict(previous_action) if previous_action else None,
                "observation": dict(previous_observation),
            })

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
        previous_action = dict(action) if action else None
        previous_observation = dict(_mapping(step.get("observation")))
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
            schema = str(raw.get("schema_version", ""))
            if schema not in {
                RECURRENT_TRANSITION_FORMAT,
                LEGACY_RECURRENT_TRANSITION_FORMAT,
            }:
                raise ValueError(
                    f"transition line {line_number} has wrong schema_version"
                )
            rows.append(
                RecurrentTransition(
                    episode_id=str(raw["episode_id"]),
                    step_index=int(raw["step_index"]),
                    goal=str(raw["goal"]),
                    event=normalize_event(dict(_mapping(raw.get("event")))),
                    target_action=str(raw["target_action"]),
                    stop_target=bool(raw["stop_target"]),
                    value_target=(
                        None if raw.get("value_target") is None else float(raw["value_target"])
                    ),
                    metadata={
                        **dict(_mapping(raw.get("metadata"))),
                        "source_transition_format": schema,
                    },
                )
            )
    return tuple(rows)


def dump_transitions(rows: Iterable[RecurrentTransition], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row.to_dict())
            payload["event"] = normalize_event(row.event)
            payload["schema_version"] = RECURRENT_TRANSITION_FORMAT
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_transition_file(episodes: Sequence[str | Path], output: str | Path) -> int:
    rows: list[RecurrentTransition] = []
    for raw_path in episodes:
        path = Path(raw_path)
        if path.is_dir():
            for child in sorted(path.glob("*.json")):
                episode = load_episode(child)
                if str(episode.get("schema_version", "")) != "lcfa.semantic-trajectory.v1":
                    continue
                rows.extend(episode_to_transitions(episode))
            continue
        rows.extend(episode_to_transitions(load_episode(path)))
    return dump_transitions(rows, output)


__all__ = [
    "ACTION_VOCAB",
    "LEGACY_RECURRENT_TRANSITION_FORMAT",
    "MAX_RECURRENT_SEQUENCE_ITEMS",
    "MAX_RECURRENT_TEXT_CHARS",
    "RECURRENT_TRANSITION_FORMAT",
    "RecurrentTransition",
    "compact_action",
    "compact_observation",
    "dump_transitions",
    "episode_to_transitions",
    "load_episode",
    "load_transitions",
    "normalize_event",
    "prepare_transition_file",
]
