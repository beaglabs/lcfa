"""Runtime repair bridge for phase-forced hybrid mutations.

The learned phase controller may correctly select a mutation while the narrow
RepairIR renderer fails because recurrent controller events intentionally omit
large source blobs. This module keeps those concerns separate: recurrent state
continues to see the compact event schema, while the mutation compiler receives
bounded source text captured from actual ``repo.read`` observations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hybrid_latent import repair_prompt
from .repair_phase import EDIT_ACTIONS
from .rwkv_controller import _json_object, _valid_generated_inputs

MAX_REPAIR_SOURCE_CHARS = 12000


def _walk_mappings(value: Any, *, depth: int = 0) -> tuple[Mapping[str, Any], ...]:
    if depth > 8:
        return ()
    out: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        out.append(value)
        for nested in value.values():
            out.extend(_walk_mappings(nested, depth=depth + 1))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value[:64]:
            out.extend(_walk_mappings(nested, depth=depth + 1))
    return tuple(out)


def collect_file_evidence(observation: Mapping[str, Any] | None) -> Mapping[str, str]:
    """Extract bounded path->text evidence from nested workspace observations."""
    if not isinstance(observation, Mapping):
        return {}
    out: dict[str, str] = {}
    for item in _walk_mappings(observation):
        path = item.get("path")
        text = item.get("text")
        if not isinstance(path, str) or not path or not isinstance(text, str):
            continue
        out[path] = text[:MAX_REPAIR_SOURCE_CHARS]
    return out


def controller_target_path(controller: Mapping[str, Any]) -> str | None:
    ir = controller.get("repair_ir")
    if isinstance(ir, Mapping):
        target = str(ir.get("target_path") or "").strip()
        if target:
            return target
    path = str(controller.get("retrieval_path") or "").strip()
    return path or None


def validate_replace_inputs(
    inputs: Mapping[str, Any] | None,
    *,
    target_path: str,
    source_text: str,
) -> Mapping[str, Any] | None:
    """Require a single exact grounded replacement before workspace execution."""
    if not isinstance(inputs, Mapping):
        return None
    candidate = dict(inputs)
    candidate["path"] = target_path
    old = candidate.get("old")
    new = candidate.get("new")
    if not isinstance(old, str) or not old or not isinstance(new, str):
        return None
    if old == new or source_text.count(old) != 1:
        return None
    if not _valid_generated_inputs("repo.replace", candidate):
        return None
    return candidate


def attach_controller_trace(
    episode_path: str | Path,
    trace: Sequence[Mapping[str, Any]],
) -> int:
    """Persist runtime controller diagnostics beside each semantic agent step."""
    path = Path(episode_path)
    if not path.is_file() or not trace:
        return 0
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        return 0
    steps = raw.get("steps")
    if not isinstance(steps, list):
        return 0
    attached = 0
    for index, controller in enumerate(trace):
        if index >= len(steps):
            break
        step = steps[index]
        if not isinstance(step, dict):
            continue
        step["controller"] = dict(controller)
        attached += 1
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return attached


class PhaseMutationRuntimePolicy:
    """Wrap a hybrid policy and rescue phase-forced mutations from search fallback."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.metadata = {
            **dict(getattr(inner, "metadata", {})),
            "phase_mutation_recovery": "grounded-source-repo.replace-v1",
            "controller_trace_persisted": True,
        }
        self._file_evidence: dict[str, str] = {}
        self._controller_trace: list[Mapping[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def reset(self, goal: str, solution: Any) -> None:
        self._file_evidence = {}
        self._controller_trace = []
        self.inner.reset(goal, solution)

    def observe(
        self,
        action: Mapping[str, Any] | None,
        observation: Mapping[str, Any],
        solution: Any,
    ) -> None:
        self._file_evidence.update(collect_file_evidence(observation))
        self.inner.observe(action, observation, solution)

    def consume_controller_trace(self) -> tuple[Mapping[str, Any], ...]:
        trace = tuple(dict(item) for item in self._controller_trace)
        self._controller_trace = []
        return trace

    def _recover_mutation(
        self,
        *,
        goal: str,
        solution: Any,
        controller: Mapping[str, Any],
        target_path: str,
    ) -> Mapping[str, Any] | None:
        source = self._file_evidence.get(target_path)
        if not source:
            return None
        model = getattr(self.inner, "model", None)
        tokenizer = getattr(self.inner, "tokenizer", None)
        torch = getattr(self.inner, "_torch", None)
        device = getattr(self.inner, "device", None)
        if model is None or tokenizer is None or torch is None or device is None:
            return None

        pointer_raw = controller.get("pointer_index")
        pointer = int(pointer_raw) if isinstance(pointer_raw, int) else None
        repair_plan_fn = getattr(self.inner, "_repair_plan", None)
        if callable(repair_plan_fn):
            plan = repair_plan_fn(
                action="repo.replace",
                pointer=pointer,
                solution=solution,
            )
        else:
            plan = {
                key: controller.get(key)
                for key in (
                    "task_intent",
                    "grounded_intent",
                    "repair_ir",
                    "repair_ir_validation",
                )
            }

        prompt = repair_prompt(
            "repo.replace",
            goal,
            {"kind": "goal", "goal": goal},
            target_path=target_path,
            repair_plan=plan,
        )
        prompt += (
            "\nGrounded source evidence follows. It is authoritative.\n"
            "Return repo.replace inputs only. Copy `old` verbatim from this source, "
            "choose the smallest uniquely occurring span that can satisfy the user request, "
            "and make `new` the minimal corrected replacement. Do not rewrite the whole file.\n"
            "Required JSON schema: {\"old\": \"exact source span\", \"new\": \"replacement\"}.\n"
            f"source_path={target_path}\n"
            f"source_text={source}\n"
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        decoder = getattr(self.inner, "patch_decoder", None)
        configured_tokens = int(getattr(decoder, "max_tokens", 512) or 512)
        max_tokens = max(128, min(768, configured_tokens))
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
                pad_token_id=(getattr(tokenizer, "pad_token_id", None) or 0),
            )
        text = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        parsed = _json_object(text)
        return validate_replace_inputs(
            parsed,
            target_path=target_path,
            source_text=source,
        )

    def choose(
        self,
        goal: str,
        solution: Any,
        recent: Sequence[Mapping[str, Any]],
        step: int,
    ) -> Mapping[str, Any]:
        result = dict(self.inner.choose(goal, solution, recent, step))
        controller_raw = result.get("controller")
        controller = dict(controller_raw) if isinstance(controller_raw, Mapping) else {}
        fallback_from = str(controller.get("fallback_from") or "")
        selected = str(controller.get("phase_selected_action") or fallback_from)
        target_path = controller_target_path(controller)

        if fallback_from in EDIT_ACTIONS and selected in EDIT_ACTIONS and target_path:
            controller["mutation_recovery_attempted"] = True
            controller["mutation_recovery_target"] = target_path
            recovered = self._recover_mutation(
                goal=goal,
                solution=solution,
                controller=controller,
                target_path=target_path,
            )
            if recovered is not None:
                result["action"] = {"name": "repo.replace", "inputs": dict(recovered)}
                result["final"] = False
                controller["mutation_recovery_status"] = "rendered"
                controller["mutation_recovery_action"] = "repo.replace"
                controller.pop("fallback_from", None)
                controller.pop("fallback_reason", None)
            else:
                # Do not silently return to an unconstrained repository search.
                # Re-read the already grounded target so the next forced mutation
                # has authoritative source text available to the compiler.
                result["action"] = {"name": "repo.read", "inputs": {"path": target_path}}
                result["final"] = False
                controller["mutation_recovery_status"] = (
                    "refresh-target-evidence"
                    if target_path in self._file_evidence
                    else "needs-target-evidence"
                )
                controller["fallback_from"] = fallback_from
                controller["fallback_reason"] = (
                    "phase-forced mutation renderer failed; refreshing grounded target evidence"
                )

        result["controller"] = controller
        self._controller_trace.append(dict(controller))
        return result


__all__ = [
    "MAX_REPAIR_SOURCE_CHARS",
    "PhaseMutationRuntimePolicy",
    "attach_controller_trace",
    "collect_file_evidence",
    "controller_target_path",
    "validate_replace_inputs",
]
