"""Text-first latent-flow dataset preparation, enrichment caching, and MLX training."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from .artifact import ARTIFACT_FORMAT
from .latent_flow import LATENT_FLOW_ARCHITECTURE
from .zplug import MLXTextZPlug, Observation, ZContext, ZPlug

LATENT_DATASET_FORMAT = "lcfa.latent-dataset.v1"
LATENT_FEATURES_FORMAT = "lcfa.latent-features.v1"


@dataclass(frozen=True, slots=True)
class TextLatentExample:
    id: str
    student_text: str
    teacher_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def mask_text(text: str, *, ratio: float, rng: random.Random) -> str:
    tokens = text.split()
    if not tokens:
        return text
    count = max(1, min(len(tokens), round(len(tokens) * max(0.0, min(1.0, ratio)))))
    for index in rng.sample(range(len(tokens)), count):
        tokens[index] = "<mask>"
    return " ".join(tokens)


def examples_from_lines(lines: Iterable[str], *, mask_ratio: float = 0.25,
                        seed: int = 0) -> tuple[TextLatentExample, ...]:
    rng = random.Random(seed)
    examples: list[TextLatentExample] = []
    for index, raw in enumerate(lines):
        text = raw.strip()
        if not text:
            continue
        examples.append(TextLatentExample(
            id=f"text:{index}",
            student_text=mask_text(text, ratio=mask_ratio, rng=rng),
            teacher_text=text,
            metadata={"mask_ratio": mask_ratio},
        ))
    return tuple(examples)


def dump_examples_jsonl(examples: Sequence[TextLatentExample], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({
                "format": LATENT_DATASET_FORMAT,
                "id": example.id,
                "student_text": example.student_text,
                "teacher_text": example.teacher_text,
                "metadata": dict(example.metadata),
            }, ensure_ascii=False, sort_keys=True) + "\n")
    return output


def load_examples_jsonl(path: str | Path) -> tuple[TextLatentExample, ...]:
    examples: list[TextLatentExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("format") not in {None, LATENT_DATASET_FORMAT}:
                raise ValueError(f"unsupported latent dataset format on line {line_number}: {raw.get('format')!r}")
            examples.append(TextLatentExample(
                id=str(raw.get("id", f"row:{line_number}")),
                student_text=str(raw["student_text"]),
                teacher_text=str(raw["teacher_text"]),
                metadata=dict(raw.get("metadata", {})),
            ))
    if not examples:
        raise ValueError("latent dataset is empty")
    return tuple(examples)


def encode_examples(examples: Sequence[TextLatentExample], zplug: ZPlug,
                    output: str | Path, *, progress: Callable[[int, int, str], None] | None = None) -> Path:
    student: list[np.ndarray] = []
    teacher: list[np.ndarray] = []
    for index, example in enumerate(examples, 1):
        if progress is not None:
            progress(index, len(examples), example.id)
        s = zplug.encode(Observation(example.id + ":student", "text", example.student_text), ZContext())
        t = zplug.encode(Observation(example.id + ":teacher", "text", example.teacher_text), ZContext())
        if len(s.features) != len(t.features):
            raise ValueError(f"feature dimension changed within example {example.id}")
        student.append(np.asarray(s.features, dtype=np.float32))
        teacher.append(np.asarray(t.features, dtype=np.float32))
    student_array = np.stack(student)
    teacher_array = np.stack(teacher)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"student": student_array, "teacher": teacher_array},
        str(output_path),
        metadata={
            "format": LATENT_FEATURES_FORMAT,
            "zplug_id": zplug.manifest.id,
            "zplug_version": zplug.manifest.version,
            "count": str(len(examples)),
            "feature_dim": str(student_array.shape[1]),
        },
    )
    ids_path = output_path.with_suffix(output_path.suffix + ".ids.json")
    ids_path.write_text(json.dumps([e.id for e in examples], indent=2) + "\n", encoding="utf-8")
    return output_path


def _load_feature_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray, Mapping[str, str]]:
    with safe_open(str(path), framework="np", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != LATENT_FEATURES_FORMAT:
            raise ValueError(f"unsupported latent feature cache format: {metadata.get('format')!r}")
        student = np.asarray(handle.get_tensor("student"), dtype=np.float32)
        teacher = np.asarray(handle.get_tensor("teacher"), dtype=np.float32)
    if student.ndim != 2 or teacher.shape != student.shape:
        raise ValueError("student/teacher feature matrices must have identical [N,D] shapes")
    return student, teacher, metadata


def _mlx_batch_indices(mx: Any, indices: Sequence[int] | np.ndarray) -> Any:
    """Convert host-side shuffled indices into the MLX integer index type.

    MLX supports advanced indexing with MLX arrays. Passing a NumPy integer
    ndarray directly is not portable across MLX/Python versions, so normalize
    through a plain Python list and construct an explicit int32 MLX array.
    """
    values = np.asarray(indices, dtype=np.int32).reshape(-1)
    return mx.array(values.tolist(), dtype=mx.int32)


def _variance_hinge_loss(ops: Any, representation: Any, *, epsilon: float = 1e-4) -> Any:
    """VICReg-style variance hinge on the pre-normalized encoder representation.

    `ops` is MLX at runtime and NumPy in contract tests. Keeping this small
    primitive backend-agnostic makes the anti-collapse geometry directly testable
    without requiring Apple MLX in Linux CI.
    """
    std = ops.sqrt(ops.var(representation, axis=0) + epsilon)
    return ops.mean(ops.maximum(ops.array(0.0), ops.array(1.0) - std))


def train_mlx_latent_predictor(
    feature_cache: str | Path,
    output_dir: str | Path,
    *,
    latent_dim: int = 256,
    hidden_dim: int = 512,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    ema_decay: float = 0.99,
    variance_weight: float = 0.05,
    seed: int = 0,
    zplug_type: str = "mlx-text",
    zplug_model_path: str | None = None,
    zplug_max_tokens: int = 2048,
    progress: Callable[[int, int, int, float], None] | None = None,
) -> Path:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten, tree_map
    except ImportError as exc:
        raise RuntimeError("latent training requires `pip install -e '.[mlx]'`") from exc

    student_np, teacher_np, cache_metadata = _load_feature_cache(feature_cache)
    feature_dim = int(student_np.shape[1])
    latent_dim = int(latent_dim)
    hidden_dim = int(hidden_dim)
    if latent_dim < 8 or hidden_dim < 8:
        raise ValueError("latent_dim and hidden_dim must be >= 8")
    mx.random.seed(int(seed))
    np.random.seed(int(seed))

    def normalize(x):
        return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), mx.array(1e-6))

    class LatentPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(feature_dim, latent_dim)
            self.dynamics = [nn.Linear(latent_dim, hidden_dim), nn.Linear(hidden_dim, latent_dim)]
            self.target_encoder = nn.Linear(feature_dim, latent_dim)
            self.target_encoder.update(self.encoder.parameters())
            self.target_encoder.freeze()

        def student_representation(self, x):
            """Return the pre-normalized LCFA encoder representation h."""
            return self.encoder(x)

        def student_latent(self, x):
            """Return unit-normalized latent z used by prediction/dynamics."""
            return normalize(self.student_representation(x))

        def predict(self, z):
            return normalize(self.dynamics[1](nn.gelu_approx(self.dynamics[0](z))))

        def target_latent(self, x):
            return normalize(self.target_encoder(x))

        def __call__(self, x):
            representation = self.student_representation(x)
            z = normalize(representation)
            return representation, z, self.predict(z)

    model = LatentPredictor()
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    def loss_components(model, student_x, teacher_x):
        representation, _z, prediction = model(student_x)
        target = mx.stop_gradient(model.target_latent(teacher_x))
        prediction_loss = mx.mean(mx.square(prediction - target))

        # Anti-collapse acts on the encoder's unconstrained representation h,
        # not on the unit-normalized prediction latent z. Applying a unit-std
        # hinge to z is mathematically incompatible with ||z||_2 = 1 and
        # creates a dimension-dependent artificial loss floor.
        variance_loss = _variance_hinge_loss(mx, representation)
        total_loss = prediction_loss + float(variance_weight) * variance_loss
        return total_loss, prediction_loss, variance_loss

    def loss_fn(model, student_x, teacher_x):
        total_loss, _prediction_loss, _variance_loss = loss_components(model, student_x, teacher_x)
        return total_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    student = mx.array(student_np)
    teacher = mx.array(teacher_np)
    count = int(student_np.shape[0])
    global_step = 0
    last_loss = 0.0
    effective_batch_size = max(1, int(batch_size))
    for epoch in range(max(1, int(epochs))):
        permutation = np.random.permutation(count)
        for start in range(0, count, effective_batch_size):
            host_indices = permutation[start:start + effective_batch_size]
            batch_indices = _mlx_batch_indices(mx, host_indices)
            xb = student[batch_indices]
            yb = teacher[batch_indices]
            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            target_params = model.target_encoder.parameters()
            encoder_params = model.encoder.parameters()
            updated_target = tree_map(
                lambda target, source: float(ema_decay) * target + (1.0 - float(ema_decay)) * source,
                target_params,
                encoder_params,
            )
            model.target_encoder.update(updated_target)
            mx.eval(model.parameters(), optimizer.state, loss)
            global_step += 1
            last_loss = float(loss.item())
            if progress is not None:
                progress(epoch + 1, max(1, int(epochs)), global_step, last_loss)

    final_total, final_prediction, final_variance = loss_components(model, student, teacher)
    mx.eval(final_total, final_prediction, final_variance)
    final_loss = float(final_total.item())
    final_prediction_loss = float(final_prediction.item())
    final_variance_loss = float(final_variance.item())

    flat = tree_flatten(model.parameters(), destination={})
    required = (
        "encoder.weight", "encoder.bias",
        "dynamics.0.weight", "dynamics.0.bias",
        "dynamics.1.weight", "dynamics.1.bias",
    )
    missing = [name for name in required if name not in flat]
    if missing:
        raise RuntimeError(f"MLX latent model did not expose expected parameters: {missing}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / "model.safetensors"
    mx.save_safetensors(
        weights_path,
        {name: flat[name] for name in required},
        metadata={
            "format": ARTIFACT_FORMAT,
            "architecture": LATENT_FLOW_ARCHITECTURE,
            "id": out.name,
            "feature_format": LATENT_FEATURES_FORMAT,
        },
    )
    digest = sha256(weights_path.read_bytes()).hexdigest()
    tensor_shapes = {
        name: {"dtype": "F32", "shape": list(flat[name].shape)}
        for name in required
    }
    manifest = {
        "format": ARTIFACT_FORMAT,
        "id": out.name,
        "version": "0.1.0-trained",
        "architecture": LATENT_FLOW_ARCHITECTURE,
        "base_backend": "lcfa-zero",
        "weights": {"file": "model.safetensors", "format": "safetensors", "sha256": digest},
        "tensors": tensor_shapes,
        "config": {
            "zplug": {
                "type": zplug_type,
                "model_path": zplug_model_path,
                "max_tokens": int(zplug_max_tokens),
                "pool": "last",
                "normalize": True,
                "feature_dim": feature_dim,
            },
            "flow": {"steps": 4, "include_vector": False},
        },
        "metadata": {
            "training": {
                "objective": "masked-latent-prediction+ema-target+pre-norm-variance",
                "prediction_space": "unit-normalized-latent",
                "anti_collapse_space": "pre-normalized-encoder-representation",
                "feature_cache": str(feature_cache),
                "feature_cache_zplug": cache_metadata.get("zplug_id"),
                "examples": count,
                "feature_dim": feature_dim,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "ema_decay": float(ema_decay),
                "variance_weight": float(variance_weight),
                "steps": global_step,
                "final_loss": final_loss,
                "final_prediction_loss": final_prediction_loss,
                "final_variance_loss": final_variance_loss,
                "last_minibatch_loss": last_loss,
                "seed": int(seed),
            },
            "solution_decoder": "disabled-until-evaluated",
        },
    }
    (out / "artifact.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def make_mlx_text_zplug(model_path: str | Path, *, max_tokens: int = 2048) -> MLXTextZPlug:
    return MLXTextZPlug(model_path, max_tokens=max_tokens, pool="last", normalize=True)


__all__ = [
    "LATENT_DATASET_FORMAT", "LATENT_FEATURES_FORMAT", "TextLatentExample", "dump_examples_jsonl",
    "encode_examples", "examples_from_lines", "load_examples_jsonl", "make_mlx_text_zplug",
    "mask_text", "train_mlx_latent_predictor",
]
