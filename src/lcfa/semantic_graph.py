"""Persistent content-addressed semantic graph for LCFA."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Protocol

from .state import content_hash

SEMANTIC_GRAPH_FORMAT = "lcfa.semantic-graph.v1"


@dataclass(frozen=True, slots=True)
class ContentRef:
    content_hash: str
    media_type: str = "application/json"
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ConceptNode:
    id: str
    kind: str
    label: str
    content: ContentRef
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConceptEdge:
    source: str
    relation: str
    target: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConceptSnapshot:
    id: str
    root: str
    node_count: int
    edge_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SEMANTIC_GRAPH_FORMAT


class SemanticGraph(Protocol):
    def put_node(self, node_id: str, kind: str, label: str, value: Any, *, metadata: Mapping[str, Any] | None = None, media_type: str = "application/json") -> ConceptNode: ...
    def add_edge(self, source: str, relation: str, target: str, *, metadata: Mapping[str, Any] | None = None) -> ConceptEdge: ...
    def get_node(self, node_id: str) -> ConceptNode: ...
    def search(self, query: str, *, kinds: Iterable[str] | None = None, limit: int = 20) -> tuple[ConceptNode, ...]: ...
    def neighbors(self, node_id: str, *, relations: Iterable[str] | None = None, direction: str = "both", limit: int = 100) -> tuple[ConceptEdge, ...]: ...


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


class SQLiteSemanticGraph:
    """SQLite graph store with immutable BLAKE3-addressed blobs and mutable logical nodes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS blobs (
                content_hash TEXT PRIMARY KEY, media_type TEXT NOT NULL,
                payload BLOB NOT NULL, size_bytes INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL,
                content_hash TEXT NOT NULL, metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT NOT NULL, relation TEXT NOT NULL, target TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY(source, relation, target)
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
            CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
        """)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "SQLiteSemanticGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def put_blob(self, value: Any, *, media_type: str = "application/json") -> ContentRef:
        if isinstance(value, bytes):
            payload = value
            digest = content_hash({"$bytes": value.hex()})
        elif isinstance(value, str) and media_type.startswith("text/"):
            payload = value.encode("utf-8")
            digest = content_hash({"$text": value})
        else:
            payload = _json(value).encode("utf-8")
            digest = content_hash(value)
        self.db.execute(
            "INSERT OR IGNORE INTO blobs(content_hash, media_type, payload, size_bytes) VALUES (?, ?, ?, ?)",
            (digest, media_type, payload, len(payload)),
        )
        self.db.commit()
        return ContentRef(digest, media_type, len(payload))

    def get_blob(self, digest: str) -> bytes:
        row = self.db.execute("SELECT payload FROM blobs WHERE content_hash=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        return bytes(row["payload"])

    def put_node(self, node_id: str, kind: str, label: str, value: Any, *, metadata: Mapping[str, Any] | None = None, media_type: str = "application/json") -> ConceptNode:
        ref = self.put_blob(value, media_type=media_type)
        meta = dict(metadata or {})
        self.db.execute("""
            INSERT INTO nodes(id, kind, label, content_hash, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, label=excluded.label,
              content_hash=excluded.content_hash, metadata_json=excluded.metadata_json
        """, (node_id, kind, label, ref.content_hash, _json(meta)))
        self.db.commit()
        return ConceptNode(node_id, kind, label, ref, meta)

    def add_edge(self, source: str, relation: str, target: str, *, metadata: Mapping[str, Any] | None = None) -> ConceptEdge:
        meta = dict(metadata or {})
        self.db.execute("""
            INSERT INTO edges(source, relation, target, metadata_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source, relation, target) DO UPDATE SET metadata_json=excluded.metadata_json
        """, (source, relation, target, _json(meta)))
        self.db.commit()
        return ConceptEdge(source, relation, target, meta)

    def get_node(self, node_id: str) -> ConceptNode:
        row = self.db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(node_id)
        blob = self.db.execute("SELECT media_type, size_bytes FROM blobs WHERE content_hash=?", (row["content_hash"],)).fetchone()
        if blob is None:
            raise KeyError(row["content_hash"])
        return ConceptNode(
            row["id"], row["kind"], row["label"],
            ContentRef(row["content_hash"], blob["media_type"], int(blob["size_bytes"])),
            json.loads(row["metadata_json"]),
        )

    def search(self, query: str, *, kinds: Iterable[str] | None = None, limit: int = 20) -> tuple[ConceptNode, ...]:
        terms = [t.lower() for t in query.replace(".", " ").replace("_", " ").split() if t]
        kinds_t = tuple(dict.fromkeys(str(k) for k in (kinds or ())))
        where: list[str] = []
        params: list[Any] = []
        if kinds_t:
            where.append(f"kind IN ({','.join('?' for _ in kinds_t)})")
            params.extend(kinds_t)
        if terms:
            term_sql = []
            for term in terms:
                term_sql.append("(lower(label) LIKE ? OR lower(id) LIKE ? OR lower(metadata_json) LIKE ?)")
                like = f"%{term}%"
                params.extend((like, like, like))
            where.append("(" + " OR ".join(term_sql) + ")")
        sql = "SELECT id FROM nodes" + ((" WHERE " + " AND ".join(where)) if where else "") + " ORDER BY kind, label LIMIT ?"
        params.append(max(1, int(limit)))
        return tuple(self.get_node(row["id"]) for row in self.db.execute(sql, params).fetchall())

    def neighbors(self, node_id: str, *, relations: Iterable[str] | None = None, direction: str = "both", limit: int = 100) -> tuple[ConceptEdge, ...]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be one of: in, out, both")
        clauses, params = [], []
        if direction in {"out", "both"}:
            clauses.append("source=?")
            params.append(node_id)
        if direction in {"in", "both"}:
            clauses.append("target=?")
            params.append(node_id)
        where = "(" + " OR ".join(clauses) + ")"
        rels = tuple(dict.fromkeys(str(r) for r in (relations or ())))
        if rels:
            where += f" AND relation IN ({','.join('?' for _ in rels)})"
            params.extend(rels)
        params.append(max(1, int(limit)))
        rows = self.db.execute(
            f"SELECT source, relation, target, metadata_json FROM edges WHERE {where} ORDER BY relation, source, target LIMIT ?",
            params,
        ).fetchall()
        return tuple(ConceptEdge(r["source"], r["relation"], r["target"], json.loads(r["metadata_json"])) for r in rows)

    def stats(self) -> Mapping[str, int]:
        return {
            "nodes": int(self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]),
            "edges": int(self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]),
            "blobs": int(self.db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]),
        }

    def snapshot(self, root: str, *, metadata: Mapping[str, Any] | None = None) -> ConceptSnapshot:
        stats = self.stats()
        meta = dict(metadata or {})
        digest = content_hash({"root": root, **stats, "metadata": meta})
        return ConceptSnapshot(
            id=f"semantic-snapshot:{digest.removeprefix('b3:')[:24]}", root=root,
            node_count=stats["nodes"], edge_count=stats["edges"], metadata=meta,
        )


__all__ = ["SEMANTIC_GRAPH_FORMAT", "ContentRef", "ConceptNode", "ConceptEdge", "ConceptSnapshot", "SemanticGraph", "SQLiteSemanticGraph"]
