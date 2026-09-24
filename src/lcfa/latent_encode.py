"""Batched, resumable frozen-feature extraction for latent-flow training."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Callable, Sequence

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from .latent_train import LATENT_FEATURES_FORMAT, TextLatentExample
from .zplug import MLXTextZPlug, Observation, ZContext, ZPlug


def _parts_dir(output: Path) -> Path:
    return output.with_name(output.name + ".parts")


def _ids_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".ids.json")


def _encode_batch(zplug: ZPlug, observations: Sequence[Observation],
                  contexts: Sequence[ZContext]) -> tuple:
    batch_fn = getattr(zplug, "encode_batch", None)
    if callable(batch_fn):
        return tuple(batch_fn(observations, contexts))
    return tuple(zplug.encode(observation, context) for observation, context in zip(observations, contexts))


def _completed_count(parts: Path, examples: Sequence[TextLatentExample], zplug: ZPlug) -> int:
    processed = 0
    if not parts.exists():
        return 0
    for path in sorted(parts.glob("part-*.safetensors")):
        with safe_open(str(path), framework="np", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("format") != LATENT_FEATURES_FORMAT:
                raise ValueError(f"invalid latent feature checkpoint format: {path}")
            if metadata.get("zplug_id") != zplug.manifest.id:
                raise ValueError(f"latent checkpoint zplug mismatch: {path}")
            start = int(metadata.get("start", "-1"))
            count = int(metadata.get("count", "0"))
            ids = json.loads(metadata.get("ids_json", "[]"))
        if start != processed or count != len(ids):
            raise ValueError(f"non-contiguous latent feature checkpoint: {path}")
        expected = [example.id for example in examples[start:start + count]]
        if ids != expected:
            raise ValueError(f"latent checkpoint dataset mismatch: {path}")
        processed += count
    if processed > len(examples):
        raise ValueError("latent checkpoints contain more examples than the current dataset")
    return processed


def _write_part(parts: Path, *, start: int, ids: Sequence[str], student: Sequence[np.ndarray],
                teacher: Sequence[np.ndarray], zplug: ZPlug) -> None:
    if not ids:
        return
    student_array = np.stack(student).astype(np.float32, copy=False)
    teacher_array = np.stack(teacher).astype(np.float32, copy=False)
    if student_array.shape != teacher_array.shape:
        raise ValueError("student/teacher checkpoint matrices must have identical shapes")
    end = start + len(ids)
    path = parts / f"part-{start:06d}-{end:06d}.safetensors"
    temp = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {"student": student_array, "teacher": teacher_array},
        str(temp),
        metadata={
            "format": LATENT_FEATURES_FORMAT,
            "zplug_id": zplug.manifest.id,
            "zplug_version": zplug.manifest.version,
            "start": str(start),
            "count": str(len(ids)),
            "feature_dim": str(student_array.shape[1]),
            "ids_json": json.dumps(list(ids), separators=(",", ":")),
        },
    )
    temp.replace(path)


def _finalize_parts(parts: Path, output: Path, examples: Sequence[TextLatentExample], zplug: ZPlug,
                    *, batch_size: int, checkpoint_every: int) -> None:
    student_chunks: list[np.ndarray] = []
    teacher_chunks: list[np.ndarray] = []
    total = 0
    feature_dim: int | None = None
    for path in sorted(parts.glob("part-*.safetensors")):
        with safe_open(str(path), framework="np", device="cpu") as handle:
            student = np.asarray(handle.get_tensor("student"), dtype=np.float32)
            teacher = np.asarray(handle.get_tensor("teacher"), dtype=np.float32)
        if student.shape != teacher.shape or student.ndim != 2:
            raise ValueError(f"invalid latent checkpoint tensor shape: {path}")
        if feature_dim is None:
            feature_dim = int(student.shape[1])
        elif student.shape[1] != feature_dim:
            raise ValueError(f"latent checkpoint feature dimension changed: {path}")
        student_chunks.append(student)
        teacher_chunks.append(teacher)
        total += int(student.shape[0])
    if total != len(examples) or feature_dim is None:
        raise ValueError(f"latent feature checkpoints incomplete: have {total}, expected {len(examples)}")

    student_array = np.concatenate(student_chunks, axis=0)
    teacher_array = np.concatenate(teacher_chunks, axis=0)
    temp = output.with_suffix(output.suffix + ".tmp")
    save_file(
        {"student": student_array, "teacher": teacher_array},
        str(temp),
        metadata={
            "format": LATENT_FEATURES_FORMAT,
            "zplug_id": zplug.manifest.id,
            "zplug_version": zplug.manifest.version,
            "count": str(len(examples)),
            "feature_dim": str(feature_dim),
            "encode_batch_size": str(batch_size),
            "checkpoint_every": str(checkpoint_every),
            "resumable": "true",
        },
    )
    temp.replace(output)
    _ids_path(output).write_text(
        json.dumps([example.id for example in examples], indent=2) + "\n",
        encoding="utf-8",
    )


def _completed_output(output: Path, examples: Sequence[TextLatentExample], zplug: ZPlug) -> bool:
    if not output.exists():
        return False
    try:
        with safe_open(str(output), framework="np", device="cpu") as handle:
            metadata = handle.metadata() or {}
            shape = tuple(handle.get_slice("student").get_shape())
        return (
            metadata.get("format") == LATENT_FEATURES_FORMAT
            and metadata.get("zplug_id") == zplug.manifest.id
            and int(metadata.get("count", "-1")) == len(examples)
            and len(shape) == 2
            and shape[0] == len(examples)
        )
    except Exception:
        return False


def encode_examples(examples: Sequence[TextLatentExample], zplug: ZPlug,
                    output: str | Path, *, progress: Callable[[int, int, str], None] | None = None,
                    batch_size: int = 2, checkpoint_every: int = 128,
                    resume: bool = True) -> Path:
    """Encode student/teacher pairs in batches with atomic resumable checkpoints."""
    if not examples:
        raise ValueError("latent dataset is empty")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = _parts_dir(output_path)
    effective_batch = max(1, int(batch_size))
    checkpoint_every = max(effective_batch, int(checkpoint_every))

    if resume and _completed_output(output_path, examples, zplug):
        return output_path
    if not resume:
        output_path.unlink(missing_ok=True)
        _ids_path(output_path).unlink(missing_ok=True)
        shutil.rmtree(parts, ignore_errors=True)

    parts.mkdir(parents=True, exist_ok=True)
    processed = _completed_count(parts, examples, zplug) if resume else 0
    shard_start = processed
    shard_ids: list[str] = []
    shard_student: list[np.ndarray] = []
    shard_teacher: list[np.ndarray] = []

    for start in range(processed, len(examples), effective_batch):
        batch = examples[start:start + effective_batch]
        student_obs = [Observation(item.id + ":student", "text", item.student_text) for item in batch]
        teacher_obs = [Observation(item.id + ":teacher", "text", item.teacher_text) for item in batch]
        observations = tuple(student_obs + teacher_obs)
        contexts = tuple(ZContext() for _ in observations)
        packets = _encode_batch(zplug, observations, contexts)
        if len(packets) != len(observations):
            raise ValueError("zplug batch encoder returned the wrong number of packets")
        width = len(batch)
        student_packets = packets[:width]
        teacher_packets = packets[width:]

        for offset, (example, student_packet, teacher_packet) in enumerate(
            zip(batch, student_packets, teacher_packets), start=1
        ):
            if len(student_packet.features) != len(teacher_packet.features):
                raise ValueError(f"feature dimension changed within example {example.id}")
            shard_ids.append(example.id)
            shard_student.append(np.asarray(student_packet.features, dtype=np.float32))
            shard_teacher.append(np.asarray(teacher_packet.features, dtype=np.float32))
            completed = start + offset
            if progress is not None:
                progress(completed, len(examples), example.id)

        if len(shard_ids) >= checkpoint_every or start + width >= len(examples):
            _write_part(
                parts,
                start=shard_start,
                ids=shard_ids,
                student=shard_student,
                teacher=shard_teacher,
                zplug=zplug,
            )
            shard_start += len(shard_ids)
            shard_ids.clear()
            shard_student.clear()
            shard_teacher.clear()

    _finalize_parts(
        parts,
        output_path,
        examples,
        zplug,
        batch_size=effective_batch,
        checkpoint_every=checkpoint_every,
    )
    shutil.rmtree(parts, ignore_errors=True)
    return output_path


def make_mlx_text_zplug(model_path: str | Path, *, max_tokens: int = 2048,
                         pad_to: int = 32) -> MLXTextZPlug:
    return MLXTextZPlug(model_path, max_tokens=max_tokens, pool="last", normalize=True, pad_to=pad_to)


__all__ = ["encode_examples", "make_mlx_text_zplug"]
