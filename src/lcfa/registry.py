"""Registries for pure reasoning operators and side-effecting actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .protocol import ActionResult, ExecutionContext, OperatorResult

OperatorHandler = Callable[[ExecutionContext, Mapping[str, object], Mapping[str, object]], OperatorResult]
ActionHandler = Callable[[ExecutionContext, Mapping[str, object]], ActionResult]


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    handler: OperatorHandler
    deterministic: bool = True
    evidence_preserving: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    handler: ActionHandler
    effects: tuple[str, ...] = ()
    description: str = ""


class OperatorRegistry:
    def __init__(self) -> None:
        self._items: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        if spec.name in self._items:
            raise ValueError(f"operator already registered: {spec.name}")
        self._items[spec.name] = spec

    def get(self, name: str) -> OperatorSpec:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"unknown operator: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


class ActionRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ActionSpec] = {}

    def register(self, spec: ActionSpec) -> None:
        if spec.name in self._items:
            raise ValueError(f"action already registered: {spec.name}")
        self._items[spec.name] = spec

    def get(self, name: str) -> ActionSpec:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"unknown action: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
