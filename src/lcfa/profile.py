"""Declarative domain profile loading for LCFA."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any, Mapping


class ProfileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Profile:
    id: str
    version: str
    description: str = ""
    entity_types: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    enabled_operators: tuple[str, ...] = ()
    enabled_actions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Profile":
        profile = data.get("profile", {})
        if not isinstance(profile, Mapping):
            raise ProfileError("[profile] must be a table")

        profile_id = str(profile.get("id", "")).strip()
        version = str(profile.get("version", "")).strip()
        if not profile_id:
            raise ProfileError("profile.id is required")
        if not version:
            raise ProfileError("profile.version is required")

        return cls(
            id=profile_id,
            version=version,
            description=str(profile.get("description", "")),
            entity_types=_string_tuple(data, "entities", "types"),
            relation_types=_string_tuple(data, "relations", "types"),
            enabled_operators=_string_tuple(data, "operators", "enable"),
            enabled_actions=_string_tuple(data, "actions", "enable"),
            metadata=dict(data.get("metadata", {})),
        )


def _string_tuple(data: Mapping[str, Any], table: str, key: str) -> tuple[str, ...]:
    section = data.get(table, {})
    if not isinstance(section, Mapping):
        raise ProfileError(f"[{table}] must be a table")
    values = section.get(key, ())
    if not isinstance(values, (list, tuple)):
        raise ProfileError(f"{table}.{key} must be an array")
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ProfileError(f"{table}.{key} may not contain empty values")
    return result


def load_profile(path: str | Path) -> Profile:
    location = Path(path)
    if location.is_dir():
        location = location / "profile.toml"
    if not location.exists():
        raise ProfileError(f"profile not found: {location}")
    with location.open("rb") as handle:
        data = tomllib.load(handle)
    return Profile.from_mapping(data)
