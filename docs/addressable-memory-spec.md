# LCFA Addressable Memory Specification (LAMS)

**Version:** 0.1  
**Schema identifier:** `lcfa.addressable-memory.v0.1`

LAMS defines LCFA's durable external cognition as a transactional, content-addressed address space. It is intentionally not a POSIX filesystem clone. Filesystem-like names are used because hierarchical addresses are convenient for agents, while canonical identity, provenance, graph relations, and retrieval remain first-class.

## Design principles

1. **Immutable objects are canonical.** Object identity is a BLAKE3 content hash over kind, media type, immutable metadata, and payload.
2. **Addresses are references.** `mem://...`, `file://...`, `symbol://...`, `identifier://...`, and related addresses point to immutable objects or world-model concepts.
3. **Indexes are disposable.** FTS and FAISS accelerate recall but are never canonical identity and may be rebuilt.
4. **Relations are typed evidence.** Code, semantic, episodic, and procedural concepts share an explicit graph edge model.
5. **Episodes are append-only.** Recurrent rollouts can be reproduced from durable ordered events.
6. **Working memory is copy-on-write.** Session or reasoning-branch overlays shadow a persistent base and use whiteouts for deletion.
7. **Snapshots are reproducible.** A snapshot root hashes the visible address-to-object mapping.

## Address space

LAMS v0.1 reserves these URI-like schemes:

- `mem://` — durable or working cognitive namespace.
- `repo://` — repository identity.
- `file://` — repository-relative file concept.
- `module://` — language module concept.
- `symbol://` — function, method, class, field, or other symbol.
- `identifier://` — exact identifier or qualified name.
- `ast://` — structural syntax occurrence or node.
- `span://` — source span.
- `test://` — test concept.
- `package://` — dependency/package concept.

Examples:

```text
mem://sessions/01JXYZ/working/goal
mem://tasks/loader-fix/attempts/3/verifier
file://src/lcfa/recurrent_train.py
symbol://python/function/lcfa.recurrent_train/train_rwkv_heads
identifier://python/AutoTokenizer.from_pretrained
identifier://python/trust_remote_code
ast://call/src/lcfa/recurrent_train.py/83/AutoTokenizer.from_pretrained
```

`mem://` paths MUST reject `.` and `..` components. An implementation MUST NOT use FAISS integer IDs, SQLite row IDs, or transient filesystem paths as canonical LAMS addresses.

## Canonical objects

```sql
CREATE TABLE mem_object (
  hash TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  media_type TEXT NOT NULL,
  payload BLOB NOT NULL,
  metadata_json TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);
```

Objects are immutable. Re-inserting the same canonical object is idempotent. Mutable concepts are represented by rebinding a logical address to a new object.

Recommended memory kinds include `working`, `episodic-event`, `semantic`, `procedural`, `observation`, `assertion`, `artifact`, `failure`, and `strategy`.

## References and namespaces

```sql
CREATE TABLE mem_ref (
  overlay_id TEXT NOT NULL,
  address TEXT NOT NULL,
  object_hash TEXT NOT NULL,
  scope TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY(overlay_id, address)
);
```

A reference resolves a logical address to an immutable object. Multiple addresses MAY point at the same object. `scope` is application-defined; typical values are `global`, `repo`, `task`, `session`, and `branch`.

## Typed relations

```sql
CREATE TABLE mem_edge (
  source TEXT NOT NULL,
  relation TEXT NOT NULL,
  target TEXT NOT NULL,
  evidence_hash TEXT,
  confidence REAL NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY(source, relation, target)
);
```

Suggested relation vocabulary:

- Repository: `contains`, `defines`, `defined_in`, `belongs_to_module`, `depends_on`.
- Code: `calls`, `references`, `imports`, `inherits`, `implements`, `overrides`, `accepts_type`, `returns_type`.
- AST: `ast_parent`, `ast_child`, `argument_of`, `keyword_of`, `condition_of`, `body_of`.
- Testing: `tested_by`, `tests`, `asserts_about`, `fixture_for`.
- Cognition: `supports`, `contradicts`, `derived_from`, `corrected_by`, `concerns`.
- Runtime: `observed_in`, `modified_by`, `failed_after`, `passed_after`.

Deterministic parser/AST edges SHOULD use confidence `1.0`. Fuzzy or learned relations SHOULD include their source and confidence in metadata.

## Episodic event log

```sql
CREATE TABLE mem_event (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  event_hash TEXT NOT NULL,
  state_hash TEXT,
  overlay_id TEXT NOT NULL,
  timestamp INTEGER NOT NULL,
  UNIQUE(episode_id, sequence)
);
```

The event payload itself is a canonical `mem_object`. The event table is an ordered index over those objects. Events SHOULD NOT be updated in place.

This supports exact reconstruction of the external event stream seen by a recurrent controller:

```text
goal -> action -> observation -> action -> verifier -> recovery -> stop
```

## Copy-on-write overlays

```sql
CREATE TABLE mem_overlay (
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  created_at INTEGER NOT NULL,
  metadata_json TEXT NOT NULL
);

CREATE TABLE mem_whiteout (
  overlay_id TEXT NOT NULL,
  address TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY(overlay_id, address)
);
```

`base` is the persistent root overlay. Lookup semantics are:

1. Check the requested overlay for a local ref.
2. If a whiteout exists, return not found.
3. Otherwise walk to the parent overlay.
4. Stop at `base`.

This allows session and reasoning branches without copying the full memory graph:

```text
base learned memory
  └─ session overlay
       ├─ candidate branch A
       └─ candidate branch B
```

Committing an overlay applies its refs and deletions to its parent. Discarding an overlay removes only that overlay's mutable namespace state; canonical objects remain deduplicated and may later be garbage-collected by policy.

## Snapshots

```sql
CREATE TABLE mem_snapshot (
  id TEXT PRIMARY KEY,
  overlay_id TEXT NOT NULL,
  parent_id TEXT,
  root_hash TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  metadata_json TEXT NOT NULL
);
```

The snapshot root is the BLAKE3 hash of the sorted visible `(address, object_hash, scope)` mapping. Snapshot identity therefore changes when the visible cognitive namespace changes, without copying object payloads.

## Lexical and exact retrieval

Implementations SHOULD maintain rebuildable exact/lexical indexes over canonical objects. The reference implementation uses SQLite FTS5 when available and falls back to bounded lexical scanning.

Exact repository localization SHOULD preserve identifiers as explicit addresses, for example:

```text
identifier://python/trust_remote_code
identifier://python/AutoTokenizer.from_pretrained
```

An AST index or ripgrep-style scan may resolve occurrences to the same canonical identifier/symbol/file addresses. Retrieval mechanisms MUST NOT create competing identity systems for the same concept.

## FAISS vector index

FAISS is a derived semantic-retrieval sidecar. SQLite stores only the durable mapping between a vector ID and canonical object hash:

```sql
CREATE TABLE mem_vector (
  vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
  object_hash TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  UNIQUE(object_hash, embedding_model)
);
```

The reference implementation uses normalized vectors with `IndexFlatIP`, making inner product equivalent to cosine similarity. A FAISS index MAY be deleted and rebuilt without loss of canonical memory.

Different embedding models MAY coexist. Changing an embedding model MUST NOT change object identity.

## Retrieval fusion

LCFA SHOULD treat retrieval as evidence fusion rather than a single search API:

```text
issue / goal
   ├─ exact identifier / AST / FTS
   ├─ FAISS semantic retrieval
   └─ graph-neighbor expansion
             ↓
       ranked candidates
             ↓
       recurrent controller
```

Reciprocal-rank fusion is preferred initially because lexical, graph, and vector scores are not naturally calibrated to the same numeric range.

## Recurrent-controller contract

The recurrent controller SHOULD consume compact LAMS addresses and evidence rather than repeated raw files. Example observation:

```json
{
  "candidates": [
    {
      "address": "file://src/lcfa/recurrent_train.py",
      "rank": 1,
      "evidence": [
        "identifier://python/trust_remote_code",
        "symbol://python/function/lcfa.recurrent_train/train_rwkv_heads"
      ]
    }
  ]
}
```

Typed actions should increasingly target addresses rather than spelling paths or regenerating entire files. A later controller version may use actions such as:

```text
memory.get
memory.put
memory.link
memory.search
memory.snapshot
memory.branch
memory.commit
```

and code transforms such as `code.add_argument` or `code.replace_expression` against `ast://` or `symbol://` targets.

LAMS v0.1 does **not** add these actions to the current recurrent `ACTION_VOCAB`; doing so would invalidate existing controller head dimensions. They are reserved for the next controller schema version.

## Consistency requirements

1. Canonical objects MUST be immutable.
2. Every `mem_ref.object_hash` MUST reference an existing object.
3. Every overlay except `base` MUST have an existing parent.
4. Overlay ancestry MUST be acyclic.
5. A local ref shadows a parent ref at the same address.
6. A whiteout hides the address from all ancestors until rebound.
7. Event sequence numbers MUST be unique within an episode.
8. Edge confidence MUST be in `[0, 1]`.
9. FAISS and FTS state MUST be treated as rebuildable acceleration data.
10. Snapshot roots MUST depend only on the visible address mapping and scopes, not transient database row IDs.

## Reference implementation

`lcfa.addressable_memory.SQLiteAddressableMemory` implements the canonical store, refs, overlays, typed graph relations, event log, snapshots, vector bindings, and FTS-backed text retrieval.

`lcfa.addressable_memory.FaissMemoryIndex` implements the optional FAISS sidecar and is available with:

```bash
pip install -e '.[memory]'
```

`lcfa.memory_actions.register_memory_actions()` exposes the memory store as a separate action registry without changing the current recurrent controller action vocabulary.
