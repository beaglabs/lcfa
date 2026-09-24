from __future__ import annotations

import numpy as np
import pytest
from safetensors import safe_open

from lcfa.latent_encode import _dataset_fingerprint, _write_part, encode_examples
from lcfa.latent_train import TextLatentExample
from lcfa.zplug import LatentPacket, Observation, ZContext, ZPlugManifest, _rounded_pad_length


class _BatchPlug:
    def __init__(self, *, pad_to: int = 32) -> None:
        self.calls: list[int] = []
        self.manifest = ZPlugManifest(
            id="test.batch",
            version="1.0.0",
            modalities=("text",),
            capabilities=("encode:text", "encode-batch:text"),
            metadata={"pad_to": pad_to, "max_tokens": 1024},
        )

    def _packet(self, observation: Observation) -> LatentPacket:
        value = float(len(str(observation.payload)))
        return LatentPacket(
            id=observation.id + ":latent",
            zplug_id=self.manifest.id,
            modality="text",
            features=(value, value + 1.0, value + 2.0),
        )

    def encode(self, observation: Observation, context: ZContext) -> LatentPacket:
        self.calls.append(1)
        return self._packet(observation)

    def encode_batch(self, observations, contexts):
        self.calls.append(len(observations))
        return tuple(self._packet(item) for item in observations)


def _examples(count: int = 5):
    return tuple(
        TextLatentExample(
            id=f"text:{i}",
            student_text=f"masked {i}",
            teacher_text=f"teacher full {i}",
        )
        for i in range(count)
    )


def test_rounded_pad_length_uses_fixed_buckets_and_maximum() -> None:
    assert _rounded_pad_length(1, pad_to=32, max_tokens=1024) == 32
    assert _rounded_pad_length(33, pad_to=32, max_tokens=1024) == 64
    assert _rounded_pad_length(1024, pad_to=32, max_tokens=1024) == 1024
    assert _rounded_pad_length(1200, pad_to=32, max_tokens=1024) == 1024


def test_encode_examples_batches_student_and_teacher_and_finalizes(tmp_path) -> None:
    plug = _BatchPlug()
    output = tmp_path / "features.safetensors"
    examples = _examples(5)

    result = encode_examples(
        examples,
        plug,
        output,
        batch_size=2,
        checkpoint_every=2,
    )

    assert result == output
    assert plug.calls == [4, 4, 2]
    assert not (tmp_path / "features.safetensors.parts").exists()
    with safe_open(str(output), framework="np", device="cpu") as handle:
        metadata = handle.metadata()
        student = np.asarray(handle.get_tensor("student"))
        teacher = np.asarray(handle.get_tensor("teacher"))
    assert metadata["count"] == "5"
    assert metadata["encode_batch_size"] == "2"
    assert metadata["zplug_fingerprint"]
    assert metadata["dataset_fingerprint"] == _dataset_fingerprint(examples)
    assert student.shape == (5, 3)
    assert teacher.shape == (5, 3)
    assert np.all(teacher[:, 0] > student[:, 0])


def test_completed_feature_cache_is_reused_without_reencoding(tmp_path) -> None:
    output = tmp_path / "features.safetensors"
    examples = _examples(3)
    first = _BatchPlug()
    encode_examples(examples, first, output, batch_size=2, checkpoint_every=2)
    assert first.calls

    second = _BatchPlug()
    encode_examples(examples, second, output, batch_size=2, checkpoint_every=2, resume=True)
    assert second.calls == []


def test_completed_cache_is_not_reused_when_encoder_configuration_changes(tmp_path) -> None:
    output = tmp_path / "features.safetensors"
    examples = _examples(3)
    first = _BatchPlug(pad_to=32)
    encode_examples(examples, first, output, batch_size=2, checkpoint_every=2)

    second = _BatchPlug(pad_to=512)
    encode_examples(examples, second, output, batch_size=2, checkpoint_every=2, resume=True)
    assert second.calls != []


def test_completed_cache_is_not_reused_when_dataset_content_changes(tmp_path) -> None:
    output = tmp_path / "features.safetensors"
    examples = _examples(3)
    first = _BatchPlug()
    encode_examples(examples, first, output, batch_size=2, checkpoint_every=2)

    changed = list(examples)
    changed[0] = TextLatentExample(
        id=examples[0].id,
        student_text="different masked content",
        teacher_text=examples[0].teacher_text,
    )
    second = _BatchPlug()
    encode_examples(tuple(changed), second, output, batch_size=2, checkpoint_every=2, resume=True)
    assert second.calls != []


def test_partial_checkpoint_resumes_at_next_unfinished_example(tmp_path) -> None:
    output = tmp_path / "features.safetensors"
    parts = tmp_path / "features.safetensors.parts"
    parts.mkdir()
    examples = _examples(5)
    plug = _BatchPlug()

    _write_part(
        parts,
        start=0,
        ids=[examples[0].id, examples[1].id],
        student=[np.array([1, 2, 3], dtype=np.float32), np.array([4, 5, 6], dtype=np.float32)],
        teacher=[np.array([7, 8, 9], dtype=np.float32), np.array([10, 11, 12], dtype=np.float32)],
        zplug=plug,
        dataset_fingerprint=_dataset_fingerprint(examples),
    )

    encode_examples(examples, plug, output, batch_size=2, checkpoint_every=2, resume=True)

    # Only examples 2-4 are encoded: 2 examples => 4 sequences, then 1 => 2.
    assert plug.calls == [4, 2]
    with safe_open(str(output), framework="np", device="cpu") as handle:
        student = np.asarray(handle.get_tensor("student"))
    assert student.shape == (5, 3)
    assert np.allclose(student[0], [1, 2, 3])
    assert np.allclose(student[1], [4, 5, 6])


def test_partial_checkpoint_rejects_changed_encoder_configuration(tmp_path) -> None:
    output = tmp_path / "features.safetensors"
    parts = tmp_path / "features.safetensors.parts"
    parts.mkdir()
    examples = _examples(3)
    original = _BatchPlug(pad_to=32)

    _write_part(
        parts,
        start=0,
        ids=[examples[0].id],
        student=[np.array([1, 2, 3], dtype=np.float32)],
        teacher=[np.array([4, 5, 6], dtype=np.float32)],
        zplug=original,
        dataset_fingerprint=_dataset_fingerprint(examples),
    )

    changed = _BatchPlug(pad_to=512)
    with pytest.raises(ValueError, match="encoder configuration mismatch"):
        encode_examples(examples, changed, output, batch_size=2, checkpoint_every=2, resume=True)
