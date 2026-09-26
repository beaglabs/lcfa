"""Configurable stochastic-flow backbones.

Backbones provide semantic proposal/verifier generation for legacy/ablation
experiments. The recurrent semantic-agent path is RWKV-only. Qwen is not a
supported first-party model family.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .artifact import ArtifactError
from .model_policy import UnsupportedModelError, reject_qwen_model


@dataclass(frozen=True, slots=True)
class BackboneSample:
    text: str
    logprob: float = 0.0


class StochasticBackbone(Protocol):
    metadata: Mapping[str, Any]

    def sample(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        branches: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None = None,
    ) -> tuple[BackboneSample, ...]: ...


class ReferenceBackbone:
    """Tiny non-SOTA backbone used only for deterministic CI integration tests."""

    metadata = {"type": "reference", "quality": "test-only"}

    def sample(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        branches: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None = None,
    ) -> tuple[BackboneSample, ...]:
        import json
        del user_prompt, temperature, top_p, max_new_tokens, seed
        if "LCFA_VERIFIER" in system_prompt:
            return (BackboneSample('{"score": 0.75}', -0.01),)
        text = json.dumps({
            "answer": "Grounded LCFA solution preserved by the reference stochastic backbone.",
            "rationale": (
                "The typed LCFA anchor is retained while the stochastic-flow path is exercised."
            ),
            "confidence": 0.75,
            "evidence_ids": [],
            "tool_requests": [],
            "final": True,
        })
        return tuple(
            BackboneSample(text, -0.05 - index * 0.01)
            for index in range(max(1, branches))
        )


class TransformersCausalBackbone:
    """Local Hugging Face causal-LM adapter for non-Qwen ablations."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device_map: str | Mapping[str, Any] = "auto",
        dtype: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        max_input_tokens: int = 16384,
    ) -> None:
        try:
            reject_qwen_model(model_path)
        except UnsupportedModelError as exc:
            raise ArtifactError(str(exc)) from exc
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ArtifactError(
                "transformers backbone requires `pip install -e '.[transformers]'`"
            ) from exc
        self._torch = torch
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_input_tokens = max(256, int(max_input_tokens))
        self.metadata = {
            "type": "transformers-local",
            "model_path": self.model_path,
            "device_map": str(device_map),
            "dtype": str(dtype),
            "local_files_only": bool(local_files_only),
        }

    def _render(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        fn = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(fn):
            try:
                return fn(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n"

    def sample(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        branches: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None = None,
    ) -> tuple[BackboneSample, ...]:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        prompt = self._render(system_prompt, user_prompt)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
        branches = max(1, int(branches))
        do_sample = temperature > 0.0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, int(max_new_tokens)),
            "num_return_sequences": branches,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(1e-5, float(temperature))
            kwargs["top_p"] = max(1e-5, min(1.0, float(top_p)))
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
            transition = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                getattr(outputs, "beam_indices", None),
                normalize_logits=True,
            )
        input_length = (
            1 if self.model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
        )
        generated = outputs.sequences[:, input_length:]
        result: list[BackboneSample] = []
        for token_row, score_row in zip(generated, transition):
            text = self.tokenizer.decode(token_row, skip_special_tokens=True)
            usable = score_row[score_row < 0]
            avg = float(usable.mean().item()) if usable.numel() else 0.0
            result.append(BackboneSample(text=text, logprob=avg))
        return tuple(result)


class MLXCausalBackbone:
    """Apple-Silicon-native non-Qwen ablation backbone through mlx-lm."""

    def __init__(self, model_path: str | Path, *, max_input_tokens: int = 8192) -> None:
        try:
            reject_qwen_model(model_path)
        except UnsupportedModelError as exc:
            raise ArtifactError(str(exc)) from exc
        try:
            import mlx.core as mx
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise ArtifactError("MLX backbone requires `pip install -e '.[mlx]'`") from exc
        self._mx = mx
        self._generate = generate
        self._make_sampler = make_sampler
        self.model_path = str(model_path)
        self.model, self.tokenizer = load(self.model_path)
        self.max_input_tokens = max(256, int(max_input_tokens))
        self.metadata = {
            "type": "mlx-local",
            "model_path": self.model_path,
            "max_input_tokens": self.max_input_tokens,
        }

    def _render(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        fn = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(fn):
            try:
                return fn(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n"

    def sample(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        branches: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None = None,
    ) -> tuple[BackboneSample, ...]:
        prompt = self._render(system_prompt, user_prompt)
        prompt = prompt[-self.max_input_tokens * 6:]
        sampler = self._make_sampler(
            temp=max(0.0, float(temperature)),
            top_p=max(1e-5, min(1.0, float(top_p))),
        )
        result: list[BackboneSample] = []
        for index in range(max(1, int(branches))):
            if seed is not None:
                self._mx.random.seed(int(seed) + index)
            text = self._generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max(1, int(max_new_tokens)),
                sampler=sampler,
                verbose=False,
            )
            result.append(BackboneSample(text=str(text), logprob=0.0))
        return tuple(result)


class LlamaCppBackbone:
    """GGUF/llama.cpp local backbone for non-Qwen ablations."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
    ) -> None:
        try:
            reject_qwen_model(model_path)
        except UnsupportedModelError as exc:
            raise ArtifactError(str(exc)) from exc
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ArtifactError(
                "llama.cpp backbone requires `pip install -e '.[llama-cpp]'`"
            ) from exc
        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
            "verbose": False,
        }
        if n_threads is not None:
            kwargs["n_threads"] = int(n_threads)
        self.model = Llama(**kwargs)
        self.model_path = str(model_path)
        self.metadata = {
            "type": "llama-cpp",
            "model_path": self.model_path,
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
        }

    def sample(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        branches: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None = None,
    ) -> tuple[BackboneSample, ...]:
        out: list[BackboneSample] = []
        for index in range(max(1, int(branches))):
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=max(0.0, float(temperature)),
                top_p=max(1e-5, min(1.0, float(top_p))),
                max_tokens=max(1, int(max_new_tokens)),
                seed=None if seed is None else int(seed) + index,
            )
            choice = response["choices"][0]
            text = choice.get("message", {}).get("content", "")
            out.append(BackboneSample(str(text), 0.0))
        return tuple(out)


def create_backbone(
    kind: str,
    model_path: str | Path | None,
    *,
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    reference_allowed: bool = False,
) -> StochasticBackbone:
    kind = str(runtime.get("backbone_type") or kind)
    raw_path = runtime.get("backbone_path") or model_path
    if kind == "reference":
        if not reference_allowed:
            raise ArtifactError(
                "reference backbone is allowed only for reference_only artifacts"
            )
        return ReferenceBackbone()
    if raw_path is None:
        raise ArtifactError(f"backbone {kind!r} requires a model path")
    try:
        reject_qwen_model(raw_path)
    except UnsupportedModelError as exc:
        raise ArtifactError(str(exc)) from exc
    if kind == "transformers-local":
        return TransformersCausalBackbone(
            raw_path,
            device_map=runtime.get("device_map", config.get("device_map", "auto")),
            dtype=str(runtime.get("dtype", config.get("dtype", "auto"))),
            local_files_only=bool(config.get("local_files_only", True)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            max_input_tokens=int(config.get("max_input_tokens", 16384)),
        )
    if kind == "mlx-local":
        return MLXCausalBackbone(
            raw_path,
            max_input_tokens=int(config.get("max_input_tokens", 8192)),
        )
    if kind == "llama-cpp":
        return LlamaCppBackbone(
            raw_path,
            n_ctx=int(runtime.get("n_ctx", config.get("n_ctx", 8192))),
            n_threads=runtime.get("n_threads", config.get("n_threads")),
            n_gpu_layers=int(
                runtime.get("n_gpu_layers", config.get("n_gpu_layers", 0))
            ),
        )
    raise ArtifactError(f"unsupported stochastic backbone type: {kind!r}")


__all__ = [
    "BackboneSample",
    "StochasticBackbone",
    "ReferenceBackbone",
    "TransformersCausalBackbone",
    "MLXCausalBackbone",
    "LlamaCppBackbone",
    "create_backbone",
]
