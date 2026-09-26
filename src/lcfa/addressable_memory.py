"""LCFA Addressable Memory Specification (LAMS) runtime.

Canonical state lives in immutable BLAKE3-addressed objects. Mutable logical
addresses, overlays, graph relations, events, snapshots, FTS, and optional
FAISS indexes are layered on top of those objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote
from uuid import uuid4

from .state import canonical_bytes, content_hash


MEMORY_SPEC_FORMAT = "lcfa.addressable-memory.v0.1"
BASE_OVERLAY = "base"
_CODE_SCHEMES = {"file", "module", "symbol", "identifier", "ast", "span", "test", "package", "repo"}
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*$")


class AddressableMemoryError(RuntimeError):
    pass


class MemoryNotFoundError(AddressableMemoryError):
    pass


class MemoryConflictError(AddressableMemoryError):
    pass


class MemoryBackendUnavailable(AddressableMemoryError):
    pass


@dataclass(frozen=True, slots=True)
class MemoryObject:
    hash: str
    kind: str
    media_type: str
    value: Any
    size_bytes: int
    created_at: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MemoryRef:
    address: str
    object_hash: str
    overlay_id: str
    scope: str
    updated_at: int


@dataclass(frozen=True, slots=True)
class MemoryEdge:
    source: str
    relation: str
    target: str
    evidence_hash: str | None
    confidence: float
    created_at: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    id: int
    episode_id: str
    sequence: int
    event_hash: str
    state_hash: str | None
    overlay_id: str
    timestamp: int


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    id: str
    overlay_id: str
    parent_id: str | None
    root_hash: str
    created_at: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VectorBinding:
    vector_id: int
    object_hash: str
    embedding_model: str
    dimensions: int
    created_at: int


@dataclass(frozen=True, slots=True)
class MemoryVectorHit:
    vector_id: int
    object_hash: str
    score: float


def _now() -> int:
    return int(time.time())


def _json(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def canonical_address(scheme: str, *parts: str) -> str:
    scheme_n = str(scheme).strip().lower()
    if not _SCHEME_RE.fullmatch(scheme_n):
        raise ValueError(f"invalid address scheme: {scheme!r}")
    values = [str(part).strip() for part in parts if str(part).strip()]
    if not values:
        raise ValueError("address requires a non-empty part")
    if scheme_n == "mem":
        clean: list[str] = []
        for part in values:
            for component in part.split("/"):
                if not component:
                    continue
                if component in {".", ".."}:
                    raise ValueError("mem addresses may not contain . or ..")
                clean.append(quote(component, safe="@._-:"))
        if not clean:
            raise ValueError("mem address path must not be empty")
        return f"mem://{'/'.join(clean)}"
    return f"{scheme_n}://" + "/".join(quote(part, safe="/:@._-") for part in values)


def normalize_address(address: str) -> str:
    value = str(address).strip()
    if "://" not in value:
        raise ValueError(f"address requires URI-like scheme: {value!r}")
    scheme, body = value.split("://", 1)
    scheme = scheme.lower()
    if not _SCHEME_RE.fullmatch(scheme) or not body:
        raise ValueError(f"invalid address: {value!r}")
    return canonical_address("mem", body) if scheme == "mem" else f"{scheme}://{body}"


def code_address(kind: str, *parts: str) -> str:
    scheme = str(kind).strip().lower()
    if scheme not in _CODE_SCHEMES:
        raise ValueError(f"unsupported code address kind: {kind!r}")
    return canonical_address(scheme, *parts)


class SQLiteAddressableMemory:
    """Persistent content-addressed cognitive address space."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS mem_config (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mem_object (
                hash TEXT PRIMARY KEY, kind TEXT NOT NULL, media_type TEXT NOT NULL,
                payload BLOB NOT NULL, metadata_json TEXT NOT NULL,
                size_bytes INTEGER NOT NULL, created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mem_overlay (
                id TEXT PRIMARY KEY, parent_id TEXT, created_at INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(parent_id) REFERENCES mem_overlay(id)
            );
            CREATE TABLE IF NOT EXISTS mem_ref (
                overlay_id TEXT NOT NULL, address TEXT NOT NULL,
                object_hash TEXT NOT NULL, scope TEXT NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY(overlay_id, address),
                FOREIGN KEY(overlay_id) REFERENCES mem_overlay(id),
                FOREIGN KEY(object_hash) REFERENCES mem_object(hash)
            );
            CREATE INDEX IF NOT EXISTS idx_mem_ref_address ON mem_ref(address);
            CREATE TABLE IF NOT EXISTS mem_whiteout (
                overlay_id TEXT NOT NULL, address TEXT NOT NULL, created_at INTEGER NOT NULL,
                PRIMARY KEY(overlay_id, address),
                FOREIGN KEY(overlay_id) REFERENCES mem_overlay(id)
            );
            CREATE TABLE IF NOT EXISTS mem_edge (
                source TEXT NOT NULL, relation TEXT NOT NULL, target TEXT NOT NULL,
                evidence_hash TEXT, confidence REAL NOT NULL, metadata_json TEXT NOT NULL,
                created_at INTEGER NOT NULL, PRIMARY KEY(source, relation, target)
            );
            CREATE INDEX IF NOT EXISTS idx_mem_edge_source ON mem_edge(source, relation);
            CREATE INDEX IF NOT EXISTS idx_mem_edge_target ON mem_edge(target, relation);
            CREATE TABLE IF NOT EXISTS mem_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
                sequence INTEGER NOT NULL, event_hash TEXT NOT NULL, state_hash TEXT,
                overlay_id TEXT NOT NULL, timestamp INTEGER NOT NULL,
                UNIQUE(episode_id, sequence),
                FOREIGN KEY(event_hash) REFERENCES mem_object(hash),
                FOREIGN KEY(overlay_id) REFERENCES mem_overlay(id)
            );
            CREATE INDEX IF NOT EXISTS idx_mem_event_episode ON mem_event(episode_id, sequence);
            CREATE TABLE IF NOT EXISTS mem_snapshot (
                id TEXT PRIMARY KEY, overlay_id TEXT NOT NULL, parent_id TEXT,
                root_hash TEXT NOT NULL, created_at INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(overlay_id) REFERENCES mem_overlay(id)
            );
            CREATE TABLE IF NOT EXISTS mem_vector (
                vector_id INTEGER PRIMARY KEY AUTOINCREMENT, object_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL, dimensions INTEGER NOT NULL,
                created_at INTEGER NOT NULL, UNIQUE(object_hash, embedding_model),
                FOREIGN KEY(object_hash) REFERENCES mem_object(hash)
            );
            CREATE INDEX IF NOT EXISTS idx_mem_vector_model ON mem_vector(embedding_model);
        """)
        self.db.execute("INSERT OR IGNORE INTO mem_config(key, value) VALUES (?, ?)", ("format", MEMORY_SPEC_FORMAT))
        self.db.execute(
            "INSERT OR IGNORE INTO mem_overlay(id, parent_id, created_at, metadata_json) VALUES (?, NULL, ?, ?)",
            (BASE_OVERLAY, _now(), _json({"kind": "base"})),
        )
        self._fts_enabled = True
        try:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(object_hash UNINDEXED, text)")
        except sqlite3.OperationalError:
            self._fts_enabled = False
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "SQLiteAddressableMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _encode(value: Any, media_type: str) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str) and media_type.startswith("text/"):
            return value.encode("utf-8")
        return canonical_bytes(value)

    @staticmethod
    def _decode(payload: bytes, media_type: str) -> Any:
        if media_type.startswith("text/"):
            return payload.decode("utf-8", errors="replace")
        if media_type == "application/json" or media_type.endswith("+json"):
            return json.loads(payload.decode("utf-8"))
        return payload

    def put_object(self, value: Any, *, kind: str, media_type: str = "application/json", metadata: Mapping[str, Any] | None = None) -> MemoryObject:
        kind_n, media_n, meta = str(kind).strip(), str(media_type).strip(), dict(metadata or {})
        if not kind_n or not media_n:
            raise ValueError("kind and media_type must not be empty")
        payload = self._encode(value, media_n)
        digest = content_hash({"kind": kind_n, "media_type": media_n, "metadata": meta, "payload": payload})
        created = _now()
        self.db.execute(
            "INSERT OR IGNORE INTO mem_object(hash, kind, media_type, payload, metadata_json, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (digest, kind_n, media_n, payload, _json(meta), len(payload), created),
        )
        if self._fts_enabled:
            self.db.execute("DELETE FROM mem_fts WHERE object_hash=?", (digest,))
            text = "" if isinstance(value, bytes) else (value if isinstance(value, str) else _json(value))
            text = (text + "\n" + " ".join(f"{k} {v}" for k, v in meta.items())).strip()
            if text:
                self.db.execute("INSERT INTO mem_fts(object_hash, text) VALUES (?, ?)", (digest, text))
        self.db.commit()
        return self.get_object(digest)

    def get_object(self, digest: str) -> MemoryObject:
        row = self.db.execute("SELECT * FROM mem_object WHERE hash=?", (str(digest),)).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"memory object not found: {digest}")
        payload = bytes(row["payload"])
        return MemoryObject(row["hash"], row["kind"], row["media_type"], self._decode(payload, row["media_type"]), int(row["size_bytes"]), int(row["created_at"]), json.loads(row["metadata_json"]))

    def create_overlay(self, *, parent_id: str = BASE_OVERLAY, overlay_id: str | None = None, metadata: Mapping[str, Any] | None = None) -> str:
        self._overlay_chain(parent_id)
        resolved = str(overlay_id or f"overlay:{uuid4()}")
        try:
            self.db.execute(
                "INSERT INTO mem_overlay(id, parent_id, created_at, metadata_json) VALUES (?, ?, ?, ?)",
                (resolved, parent_id, _now(), _json(dict(metadata or {}))),
            )
            self.db.commit()
        except sqlite3.IntegrityError as exc:
            raise MemoryConflictError(f"overlay already exists: {resolved}") from exc
        return resolved

    def _overlay_chain(self, overlay_id: str) -> tuple[str, ...]:
        chain, seen = [], set()
        current: str | None = str(overlay_id)
        while current is not None:
            if current in seen:
                raise MemoryConflictError("overlay parent cycle detected")
            seen.add(current)
            row = self.db.execute("SELECT id, parent_id FROM mem_overlay WHERE id=?", (current,)).fetchone()
            if row is None:
                raise MemoryNotFoundError(f"overlay not found: {current}")
            chain.append(row["id"])
            current = row["parent_id"]
        return tuple(chain)

    def bind(self, address: str, object_hash: str, *, overlay_id: str = BASE_OVERLAY, scope: str = "global") -> MemoryRef:
        address_n = normalize_address(address)
        self.get_object(object_hash)
        self._overlay_chain(overlay_id)
        now = _now()
        with self.db:
            self.db.execute("DELETE FROM mem_whiteout WHERE overlay_id=? AND address=?", (overlay_id, address_n))
            self.db.execute(
                """INSERT INTO mem_ref(overlay_id, address, object_hash, scope, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(overlay_id, address) DO UPDATE SET
                     object_hash=excluded.object_hash, scope=excluded.scope, updated_at=excluded.updated_at""",
                (overlay_id, address_n, object_hash, str(scope), now),
            )
        return MemoryRef(address_n, object_hash, overlay_id, str(scope), now)

    def put(self, address: str, value: Any, *, kind: str, media_type: str = "application/json", metadata: Mapping[str, Any] | None = None, overlay_id: str = BASE_OVERLAY, scope: str = "global") -> MemoryRef:
        obj = self.put_object(value, kind=kind, media_type=media_type, metadata=metadata)
        return self.bind(address, obj.hash, overlay_id=overlay_id, scope=scope)

    def resolve_ref(self, address: str, *, overlay_id: str = BASE_OVERLAY) -> MemoryRef:
        address_n = normalize_address(address)
        for layer in self._overlay_chain(overlay_id):
            row = self.db.execute("SELECT * FROM mem_ref WHERE overlay_id=? AND address=?", (layer, address_n)).fetchone()
            if row is not None:
                return MemoryRef(row["address"], row["object_hash"], row["overlay_id"], row["scope"], int(row["updated_at"]))
            if self.db.execute("SELECT 1 FROM mem_whiteout WHERE overlay_id=? AND address=?", (layer, address_n)).fetchone() is not None:
                break
        raise MemoryNotFoundError(f"memory address not found: {address_n}")

    def resolve(self, address: str, *, overlay_id: str = BASE_OVERLAY) -> MemoryObject:
        return self.get_object(self.resolve_ref(address, overlay_id=overlay_id).object_hash)

    def unbind(self, address: str, *, overlay_id: str = BASE_OVERLAY) -> None:
        address_n = normalize_address(address)
        self._overlay_chain(overlay_id)
        with self.db:
            self.db.execute("DELETE FROM mem_ref WHERE overlay_id=? AND address=?", (overlay_id, address_n))
            if overlay_id == BASE_OVERLAY:
                self.db.execute("DELETE FROM mem_whiteout WHERE overlay_id=? AND address=?", (overlay_id, address_n))
            else:
                self.db.execute(
                    """INSERT INTO mem_whiteout(overlay_id, address, created_at) VALUES (?, ?, ?)
                       ON CONFLICT(overlay_id, address) DO UPDATE SET created_at=excluded.created_at""",
                    (overlay_id, address_n, _now()),
                )

    def list_refs(self, *, prefix: str | None = None, overlay_id: str = BASE_OVERLAY) -> tuple[MemoryRef, ...]:
        visible: dict[str, MemoryRef] = {}
        for layer in reversed(self._overlay_chain(overlay_id)):
            for row in self.db.execute("SELECT address FROM mem_whiteout WHERE overlay_id=?", (layer,)).fetchall():
                visible.pop(row["address"], None)
            for row in self.db.execute("SELECT * FROM mem_ref WHERE overlay_id=?", (layer,)).fetchall():
                visible[row["address"]] = MemoryRef(row["address"], row["object_hash"], row["overlay_id"], row["scope"], int(row["updated_at"]))
        prefix_n = normalize_address(prefix) if prefix else None
        return tuple(visible[key] for key in sorted(visible) if prefix_n is None or key.startswith(prefix_n))

    def commit_overlay(self, overlay_id: str) -> str:
        if overlay_id == BASE_OVERLAY:
            raise ValueError("base overlay cannot be committed")
        row = self.db.execute("SELECT parent_id FROM mem_overlay WHERE id=?", (overlay_id,)).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"overlay not found: {overlay_id}")
        parent_id = str(row["parent_id"])
        with self.db:
            for item in self.db.execute("SELECT address FROM mem_whiteout WHERE overlay_id=?", (overlay_id,)).fetchall():
                self.db.execute("DELETE FROM mem_ref WHERE overlay_id=? AND address=?", (parent_id, item["address"]))
                if parent_id != BASE_OVERLAY:
                    self.db.execute("INSERT OR REPLACE INTO mem_whiteout(overlay_id, address, created_at) VALUES (?, ?, ?)", (parent_id, item["address"], _now()))
            for item in self.db.execute("SELECT address, object_hash, scope FROM mem_ref WHERE overlay_id=?", (overlay_id,)).fetchall():
                self.db.execute("DELETE FROM mem_whiteout WHERE overlay_id=? AND address=?", (parent_id, item["address"]))
                self.db.execute(
                    """INSERT INTO mem_ref(overlay_id, address, object_hash, scope, updated_at) VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(overlay_id, address) DO UPDATE SET object_hash=excluded.object_hash, scope=excluded.scope, updated_at=excluded.updated_at""",
                    (parent_id, item["address"], item["object_hash"], item["scope"], _now()),
                )
        return parent_id

    def discard_overlay(self, overlay_id: str) -> None:
        if overlay_id == BASE_OVERLAY:
            raise ValueError("base overlay cannot be discarded")
        child = self.db.execute("SELECT id FROM mem_overlay WHERE parent_id=? LIMIT 1", (overlay_id,)).fetchone()
        if child is not None:
            raise MemoryConflictError(f"overlay {overlay_id} has child overlay {child['id']}")
        with self.db:
            self.db.execute("DELETE FROM mem_ref WHERE overlay_id=?", (overlay_id,))
            self.db.execute("DELETE FROM mem_whiteout WHERE overlay_id=?", (overlay_id,))
            self.db.execute("DELETE FROM mem_overlay WHERE id=?", (overlay_id,))

    def link(self, source: str, relation: str, target: str, *, evidence_hash: str | None = None, confidence: float = 1.0, metadata: Mapping[str, Any] | None = None) -> MemoryEdge:
        source_n, target_n, relation_n = normalize_address(source), normalize_address(target), str(relation).strip()
        confidence_f = float(confidence)
        if not relation_n:
            raise ValueError("relation must not be empty")
        if not 0.0 <= confidence_f <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if evidence_hash is not None:
            self.get_object(evidence_hash)
        meta, created = dict(metadata or {}), _now()
        self.db.execute(
            """INSERT INTO mem_edge(source, relation, target, evidence_hash, confidence, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source, relation, target) DO UPDATE SET evidence_hash=excluded.evidence_hash, confidence=excluded.confidence, metadata_json=excluded.metadata_json, created_at=excluded.created_at""",
            (source_n, relation_n, target_n, evidence_hash, confidence_f, _json(meta), created),
        )
        self.db.commit()
        return MemoryEdge(source_n, relation_n, target_n, evidence_hash, confidence_f, created, meta)

    def neighbors(self, address: str, *, relations: Iterable[str] | None = None, direction: str = "both", limit: int = 100) -> tuple[MemoryEdge, ...]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be one of: in, out, both")
        address_n, clauses, params = normalize_address(address), [], []
        if direction in {"out", "both"}:
            clauses.append("source=?")
            params.append(address_n)
        if direction in {"in", "both"}:
            clauses.append("target=?")
            params.append(address_n)
        where = "(" + " OR ".join(clauses) + ")"
        rels = tuple(dict.fromkeys(str(item) for item in (relations or ())))
        if rels:
            where += f" AND relation IN ({','.join('?' for _ in rels)})"
            params.extend(rels)
        params.append(max(1, int(limit)))
        rows = self.db.execute(f"SELECT * FROM mem_edge WHERE {where} ORDER BY confidence DESC, relation, source, target LIMIT ?", params).fetchall()
        return tuple(MemoryEdge(row["source"], row["relation"], row["target"], row["evidence_hash"], float(row["confidence"]), int(row["created_at"]), json.loads(row["metadata_json"])) for row in rows)

    def append_event(self, episode_id: str, event: Any, *, state_hash: str | None = None, overlay_id: str = BASE_OVERLAY, sequence: int | None = None, metadata: Mapping[str, Any] | None = None) -> MemoryEvent:
        episode = str(episode_id).strip()
        if not episode:
            raise ValueError("episode_id must not be empty")
        self._overlay_chain(overlay_id)
        obj = self.put_object(event, kind="episodic-event", metadata={"episode_id": episode, **dict(metadata or {})})
        if sequence is None:
            row = self.db.execute("SELECT MAX(sequence) AS seq FROM mem_event WHERE episode_id=?", (episode,)).fetchone()
            sequence = 1 if row["seq"] is None else int(row["seq"]) + 1
        timestamp = _now()
        try:
            cur = self.db.execute(
                "INSERT INTO mem_event(episode_id, sequence, event_hash, state_hash, overlay_id, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (episode, int(sequence), obj.hash, state_hash, overlay_id, timestamp),
            )
            self.db.commit()
        except sqlite3.IntegrityError as exc:
            raise MemoryConflictError(f"duplicate event sequence {sequence} for {episode}") from exc
        return MemoryEvent(int(cur.lastrowid), episode, int(sequence), obj.hash, state_hash, overlay_id, timestamp)

    def events(self, episode_id: str) -> tuple[MemoryEvent, ...]:
        rows = self.db.execute("SELECT * FROM mem_event WHERE episode_id=? ORDER BY sequence", (str(episode_id),)).fetchall()
        return tuple(MemoryEvent(int(row["id"]), row["episode_id"], int(row["sequence"]), row["event_hash"], row["state_hash"], row["overlay_id"], int(row["timestamp"])) for row in rows)

    def snapshot(self, *, overlay_id: str = BASE_OVERLAY, metadata: Mapping[str, Any] | None = None) -> MemorySnapshot:
        refs = self.list_refs(overlay_id=overlay_id)
        root_hash = content_hash([{"address": ref.address, "object_hash": ref.object_hash, "scope": ref.scope} for ref in refs])
        previous = self.db.execute("SELECT id FROM mem_snapshot WHERE overlay_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1", (overlay_id,)).fetchone()
        parent_id = previous["id"] if previous is not None else None
        meta, created = dict(metadata or {}), _now()
        snapshot_id = "memory-snapshot:" + content_hash({"overlay": overlay_id, "parent": parent_id, "root": root_hash, "metadata": meta}).removeprefix("b3:")[:24]
        self.db.execute("INSERT OR IGNORE INTO mem_snapshot(id, overlay_id, parent_id, root_hash, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?)", (snapshot_id, overlay_id, parent_id, root_hash, created, _json(meta)))
        self.db.commit()
        return MemorySnapshot(snapshot_id, overlay_id, parent_id, root_hash, created, meta)

    def bind_vector(self, object_hash: str, embedding_model: str, dimensions: int) -> VectorBinding:
        self.get_object(object_hash)
        model, dims = str(embedding_model).strip(), int(dimensions)
        if not model or dims <= 0:
            raise ValueError("embedding_model and positive dimensions are required")
        self.db.execute("INSERT OR IGNORE INTO mem_vector(object_hash, embedding_model, dimensions, created_at) VALUES (?, ?, ?, ?)", (object_hash, model, dims, _now()))
        self.db.commit()
        row = self.db.execute("SELECT * FROM mem_vector WHERE object_hash=? AND embedding_model=?", (object_hash, model)).fetchone()
        if int(row["dimensions"]) != dims:
            raise MemoryConflictError(f"vector dimension mismatch for {object_hash}: {row['dimensions']} != {dims}")
        return VectorBinding(int(row["vector_id"]), row["object_hash"], row["embedding_model"], int(row["dimensions"]), int(row["created_at"]))

    def vector_binding_by_id(self, vector_id: int) -> VectorBinding:
        row = self.db.execute("SELECT * FROM mem_vector WHERE vector_id=?", (int(vector_id),)).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"vector binding not found: {vector_id}")
        return VectorBinding(int(row["vector_id"]), row["object_hash"], row["embedding_model"], int(row["dimensions"]), int(row["created_at"]))

    def search_text(self, query: str, *, kinds: Iterable[str] | None = None, limit: int = 20) -> tuple[MemoryObject, ...]:
        query_n = str(query).strip()
        if not query_n:
            return ()
        cap, hashes = max(1, int(limit)), []
        if self._fts_enabled:
            terms = re.findall(r"[A-Za-z0-9_:.@/-]+", query_n)
            if terms:
                match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                try:
                    hashes = [row["object_hash"] for row in self.db.execute("SELECT object_hash FROM mem_fts WHERE mem_fts MATCH ? LIMIT ?", (match, cap * 4)).fetchall()]
                except sqlite3.OperationalError:
                    hashes = []
        if not hashes:
            like = f"%{query_n.lower()}%"
            hashes = [row["hash"] for row in self.db.execute("SELECT hash FROM mem_object WHERE lower(CAST(payload AS TEXT)) LIKE ? OR lower(metadata_json) LIKE ? LIMIT ?", (like, like, cap * 4)).fetchall()]
        kinds_t, seen, out = tuple(dict.fromkeys(str(kind) for kind in (kinds or ()))), set(), []
        for digest in hashes:
            if digest in seen:
                continue
            seen.add(digest)
            obj = self.get_object(digest)
            if kinds_t and obj.kind not in kinds_t:
                continue
            out.append(obj)
            if len(out) >= cap:
                break
        return tuple(out)

    def stats(self) -> Mapping[str, int]:
        tables = ("mem_object", "mem_ref", "mem_edge", "mem_event", "mem_snapshot", "mem_overlay", "mem_vector")
        return {table.removeprefix("mem_"): int(self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}


class FaissMemoryIndex:
    """Rebuildable cosine-similarity FAISS sidecar over canonical object hashes."""

    def __init__(self, store: SQLiteAddressableMemory, *, embedding_model: str, dimensions: int) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise MemoryBackendUnavailable("FAISS memory search requires `pip install -e '.[memory]'`") from exc
        self._faiss, self._np, self.store = faiss, np, store
        self.embedding_model, self.dimensions = str(embedding_model), int(dimensions)
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimensions))

    def _vector(self, vector: Sequence[float]):
        array = self._np.asarray([list(vector)], dtype="float32")
        if array.shape != (1, self.dimensions):
            raise ValueError(f"expected vector dimension {self.dimensions}, got {array.shape}")
        self._faiss.normalize_L2(array)
        return array

    def add(self, object_hash: str, vector: Sequence[float]) -> VectorBinding:
        binding = self.store.bind_vector(object_hash, self.embedding_model, self.dimensions)
        ids = self._np.asarray([binding.vector_id], dtype="int64")
        self.index.remove_ids(ids)
        self.index.add_with_ids(self._vector(vector), ids)
        return binding

    def search(self, vector: Sequence[float], *, limit: int = 20) -> tuple[MemoryVectorHit, ...]:
        scores, ids = self.index.search(self._vector(vector), max(1, int(limit)))
        out = []
        for score, vector_id in zip(scores[0], ids[0]):
            if int(vector_id) < 0:
                continue
            binding = self.store.vector_binding_by_id(int(vector_id))
            out.append(MemoryVectorHit(int(vector_id), binding.object_hash, float(score)))
        return tuple(out)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(target))

    @classmethod
    def load(cls, store: SQLiteAddressableMemory, path: str | Path, *, embedding_model: str, dimensions: int) -> "FaissMemoryIndex":
        instance = cls(store, embedding_model=embedding_model, dimensions=dimensions)
        instance.index = instance._faiss.read_index(str(path))
        if int(instance.index.d) != int(dimensions):
            raise MemoryConflictError(f"FAISS index dimension {instance.index.d} != {dimensions}")
        return instance


__all__ = [
    "MEMORY_SPEC_FORMAT", "BASE_OVERLAY", "AddressableMemoryError",
    "MemoryNotFoundError", "MemoryConflictError", "MemoryBackendUnavailable",
    "MemoryObject", "MemoryRef", "MemoryEdge", "MemoryEvent", "MemorySnapshot",
    "VectorBinding", "MemoryVectorHit", "canonical_address", "normalize_address",
    "code_address", "SQLiteAddressableMemory", "FaissMemoryIndex",
]
