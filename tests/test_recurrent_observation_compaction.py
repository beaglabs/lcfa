from __future__ import annotations

import json

from lcfa.recurrent_transitions import (
    MAX_RECURRENT_TEXT_CHARS,
    MAX_RECURRENT_SEQUENCE_ITEMS,
    normalize_event,
)


def test_repo_read_observation_is_deduplicated_and_bounded() -> None:
    text = "head\n" + ("x" * 12000) + "\ntail"
    file_value = {
        "path": "src/example.py",
        "start_line": 1,
        "end_line": 500,
        "text": text,
    }
    event = normalize_event({
        "kind": "transition",
        "action": {"name": "repo.read", "inputs": {"path": "src/example.py"}},
        "observation": {
            "graph_id": "g",
            "results": {"node-1": file_value},
            "observations": {"file": file_value},
        },
    })

    assert set(event["observation"]) == {"file"}
    compacted_text = event["observation"]["file"]["text"]
    assert compacted_text["truncated"] is True
    assert compacted_text["chars"] == len(text)
    assert len(compacted_text["preview"]) <= MAX_RECURRENT_TEXT_CHARS + 32
    assert text not in json.dumps(event)
    assert event["observation"]["file"]["path"] == "src/example.py"


def test_search_hit_sequences_are_bounded() -> None:
    hits = [
        {"path": f"src/{index}.py", "line": index + 1, "text": "match"}
        for index in range(100)
    ]
    search_value = {"query": "match", "hits": hits}
    event = normalize_event({
        "kind": "transition",
        "action": {"name": "repo.search", "inputs": {"query": "match"}},
        "observation": {
            "results": {"node-1": search_value},
            "observations": {"search": search_value},
        },
    })

    compacted_hits = event["observation"]["search"]["hits"]
    assert compacted_hits["truncated"] is True
    assert compacted_hits["total_items"] == 100
    assert len(compacted_hits["items"]) == MAX_RECURRENT_SEQUENCE_ITEMS
