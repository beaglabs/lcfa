"""Transition datasets for training recurrent LCFA controllers.

The event stream contains only information available during rollout: the goal,
a compact shared retrieval context, and prior action/observation pairs.  Action
supervision also retains the target inputs and optional candidate-pointer label
so training covers the noun as well as the verb.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .repo_retrieval import pointer_for_action, retrieval_from_mapping


RECURRENT_TRANSITION_FORMAT = "lcfa.recurrent-transition.v3"
LEGACY_RECURRENT_TRANSITION_FORMATS = (
    "lcfa.recurrent-transition.v1",
    "lcfa.recurrent-transition.v2",
)
LEGACY_RECURRENT_TRANSITION_FORMAT = LEGACY_RECURRENT_TRANSITION_FORMATS[0]
MAX_RECURRENT_TEXT_CHARS = 4096
MAX_RECURRENT_SEQUENCE_ITEMS = 24
MAX_RETRIEVAL_QUERIES_IN_EVENT = 8
MAX_RETRIEVAL_PATHS_IN_EVENT = 16

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
    target_inputs: Mapping[str, Any] | None = None
    target_pointer: int | None = None
    candidate_queries: tuple[str, ...] = ()
    candidate_paths: tuple[str, ...] = ()
    schema_version: str = RECURRENT_TRANSITION_FORMAT

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text_summary(value: str) -> Mapping[str, Any]:
    raw = value.encode("utf-8")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


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
        return {"items": compacted, "total_items": len(items), "truncated": True}
    return value


def compact_action(action: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
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


def _named_observation(observation: Mapping[str, Any], key: str) -> Any | None:
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


def _retrieval_summary(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    raw_queries = value.get("queries", ())
    queries = (
        [str(item) for item in raw_queries[:MAX_RETRIEVAL_QUERIES_IN_EVENT]]
        if isinstance(raw_queries, Sequence) and not isinstance(raw_queries, (str, bytes))
        else []
    )
    raw_paths = value.get("paths", ())
    if not raw_paths:
        candidates = value.get("candidates", ())
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            raw_paths = [
                str(item.get("path"))
                for item in candidates
                if isinstance(item, Mapping) and item.get("path")
            ]
    paths = (
        [str(item) for item in raw_paths[:MAX_RETRIEVAL_PATHS_IN_EVENT]]
        if isinstance(raw_paths, Sequence) and not isinstance(raw_paths, (str, bytes))
        else []
    )
    return {"queries": queries, "paths": paths}


def normalize_event(event: Mapping[str, Any]) -> Mapping[str, Any]:
    kind = str(event.get("kind") or "")
    if kind == "goal":
        out: dict[str, Any] = {
            "kind": "goal",
            "goal": _compact_text(str(event.get("goal") or "")),
        }
        retrieval = _retrieval_summary(
            event.get("retrieval") if isinstance(event.get("retrieval"), Mapping) else None
        )
        if retrieval is not None:
            out["retrieval"] = retrieval
        return out
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

    episode_metadata = _mapping(episode.get("metadata"))
    retrieval_raw = _mapping(episode_metadata.get("retrieval"))
    retrieval = retrieval_from_mapping(retrieval_raw)
    candidate_queries = retrieval.queries if retrieval is not None else ()
    candidate_paths = retrieval.candidate_paths if retrieval is not None else ()
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
        inputs = dict(_mapping(action.get("inputs"))) if action else {}

        if position == 1:
            event: Mapping[str, Any] = normalize_event({
                "kind": "goal",
                "goal": goal,
                "retrieval": {
                    "queries": list(candidate_queries),
                    "paths": list(candidate_paths),
                },
            })
        else:
            event = normalize_event({
                "kind": "transition",
                "action": dict(previous_action) if previous_action else None,
                "observation": dict(previous_observation),
            })

        out.append(RecurrentTransition(
            episode_id=episode_id,
            step_index=int(step.get("index", position)),
            goal=goal,
            event=event,
            target_action=action_name,
            stop_target=terminal,
            value_target=value_target,
            metadata={"has_observation": bool(step.get("observation"))},
            target_inputs=inputs,
            target_pointer=pointer_for_action(action_name, inputs, retrieval),
            candidate_queries=tuple(candidate_queries),
            candidate_paths=tuple(candidate_paths),
        ))
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
            if schema not in {RECURRENT_TRANSITION_FORMAT, *LEGACY_RECURRENT_TRANSITION_FORMATS}:
                raise ValueError(f"transition line {line_number} has wrong schema_version")
            raw_queries = raw.get("candidate_queries", ())
            raw_paths = raw.get("candidate_paths", ())
            rows.append(RecurrentTransition(
                episode_id=str(raw["episode_id"]),
                step_index=int(raw["step_index"]),
                goal=str(raw["goal"]),
                event=normalize_event(dict(_mapping(raw.get("event")))),
                target_action=str(raw["target_action"]),
                stop_target=bool(raw["stop_target"]),
                value_target=(None if raw.get("value_target") is None else float(raw["value_target"])),
                metadata={
                    **dict(_mapping(raw.get("metadata"))),
                    "source_transition_format": schema,
                },
                target_inputs=dict(_mapping(raw.get("target_inputs"))),
                target_pointer=(None if raw.get("target_pointer") is None else int(raw["target_pointer"])),
                candidate_queries=(
                    tuple(str(item) for item in raw_queries)
                    if isinstance(raw_queries, Sequence) and not isinstance(raw_queries, (str, bytes))
                    else ()
                ),
                candidate_paths=(
                    tuple(str(item) for item in raw_paths)
                    if isinstance(raw_paths, Sequence) and not isinstance(raw_paths, (str, bytes))
                    else ()
                ),
            ))
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


def merge_transition_files(inputs: Sequence[str | Path], output: str | Path) -> int:
    rows: list[RecurrentTransition] = []
    for path in inputs:
        rows.extend(load_transitions(path))
    return dump_transitions(rows, output)


__all__ = [
    "ACTION_VOCAB",
    "LEGACY_RECURRENT_TRANSITION_FORMAT",
    "LEGACY_RECURRENT_TRANSITION_FORMATS",
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
    "merge_transition_files",
    "normalize_event",
    "prepare_transition_file",
]
