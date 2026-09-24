"""Adaptive test-time compute for ``lcfa.stochastic-flow.v1``.

The base stochastic reasoner defines the maximum branch/step/verifier budget.
This subclass starts with a smaller budget and only spends more compute when the
current candidate set is uncertain.  The public ReasoningPlan -> SolutionState
contract is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .protocol import ExecutionContext, ReasoningPlan, SolutionState
from .stochastic import FlowCandidate, StochasticFlowReasoner


@dataclass(frozen=True, slots=True)
class ComputeAssessment:
    uncertain: bool
    reasons: tuple[str, ...]
    confidence: float
    evidence_score: float
    score_margin: float | None
    verifier_score: float | None
    must_continue: bool


class AdaptiveStochasticFlowReasoner(StochasticFlowReasoner):
    """Stochastic-flow reasoner with bounded, uncertainty-triggered compute."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        flow = dict(self.artifact.config.get("flow", {}))
        adaptive = dict(flow.get("adaptive_compute", {}))
        self.adaptive_enabled = bool(adaptive.get("enabled", False))
        self.initial_branches = max(
            1,
            min(self.branches, int(adaptive.get("initial_branches", 1))),
        )
        self.adaptive_expand = bool(adaptive.get("expand_when_uncertain", True))
        self.adaptive_verify = bool(adaptive.get("verify_when_uncertain", True))
        self.confidence_threshold = float(adaptive.get("confidence_threshold", 0.80))
        self.evidence_threshold = float(adaptive.get("evidence_threshold", 0.50))
        self.score_margin_threshold = float(adaptive.get("score_margin_threshold", 0.15))
        self.verifier_accept_threshold = float(adaptive.get("verifier_accept_threshold", 0.85))
        self.require_parsed = bool(adaptive.get("require_parsed", True))
        self.metadata = {
            **dict(self.metadata),
            "adaptive_compute": {
                "enabled": self.adaptive_enabled,
                "initial_branches": self.initial_branches,
                "max_branches": self.branches,
                "max_steps": self.max_steps,
                "confidence_threshold": self.confidence_threshold,
                "evidence_threshold": self.evidence_threshold,
                "score_margin_threshold": self.score_margin_threshold,
                "verify_when_uncertain": self.adaptive_verify,
                "expand_when_uncertain": self.adaptive_expand,
            },
        }

    def _note(self, event: str, **payload: Any) -> None:
        note = getattr(self.backbone, "note", None)
        if callable(note):
            try:
                note({"event": event, **payload})
            except Exception:
                # Progress reporting must never affect reasoning correctness.
                pass

    def _assessment(
        self,
        candidates: Sequence[FlowCandidate],
        anchor: SolutionState,
    ) -> ComputeAssessment:
        if not candidates:
            return ComputeAssessment(True, ("no-candidates",), 0.0, 0.0, None, None, True)
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        reasons: list[str] = []
        verifier_accepted = (
            best.verifier_score is not None
            and best.verifier_score >= self.verifier_accept_threshold
        )
        if self.require_parsed and not best.parsed:
            reasons.append("unparsed")
        if not best.final and not best.tool_requests:
            reasons.append("not-final")
        if best.confidence < self.confidence_threshold and not verifier_accepted:
            reasons.append("low-confidence")
        evidence_score = self._evidence_score(best.evidence_ids, anchor)
        if anchor.evidence and evidence_score < self.evidence_threshold:
            reasons.append("weak-evidence")
        if any(obs.get("status") == "error" for obs in best.tool_observations):
            reasons.append("tool-error")
        margin: float | None = None
        if len(ranked) > 1:
            margin = ranked[0].score - ranked[1].score
            if margin < self.score_margin_threshold:
                reasons.append("small-score-margin")
        return ComputeAssessment(
            uncertain=bool(reasons),
            reasons=tuple(reasons),
            confidence=best.confidence,
            evidence_score=evidence_score,
            score_margin=margin,
            verifier_score=best.verifier_score,
            must_continue=bool(best.tool_requests) or not best.final,
        )

    def _sample_for_parents(
        self,
        *,
        plan: ReasoningPlan,
        context: ExecutionContext,
        anchor: SolutionState,
        beam: Sequence[FlowCandidate | None],
        step: int,
        branches: int,
        branch_offset: int = 0,
        seed_offset: int = 0,
    ) -> list[FlowCandidate]:
        proposed: list[FlowCandidate] = []
        for parent_index, parent in enumerate(beam):
            seed = (
                None
                if self.seed is None
                else self.seed + step * 1000 + parent_index * 100 + seed_offset
            )
            samples = self.backbone.sample(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=self._prompt(plan, context, anchor, parent),
                branches=branches,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                seed=seed,
            )
            proposed.extend(
                self._candidate(
                    sample,
                    step,
                    branch_offset + branch,
                    parent,
                    anchor,
                    context,
                )
                for branch, sample in enumerate(samples)
            )
        return proposed

    def _rescore_after_adaptation(
        self,
        proposed: Sequence[FlowCandidate],
        anchor: SolutionState,
    ) -> list[FlowCandidate]:
        rescored: list[FlowCandidate] = []
        for candidate in proposed:
            features = self._features(
                logprob=candidate.logprob,
                confidence=candidate.confidence,
                evidence=self._evidence_score(candidate.evidence_ids, anchor),
                tool=self._tool_score(candidate.tool_observations),
                parsed=candidate.parsed,
                final=candidate.final,
                verifier=candidate.verifier_score,
            )
            prior_score = self.fast_prior.score(features)
            base_score = (
                candidate.heuristic_score
                + self.score_weights["verifier"] * float(candidate.verifier_score or 0.0)
            )
            rescored.append(
                replace(
                    candidate,
                    prior_score=prior_score,
                    score=base_score + self.prior_weight * prior_score,
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)

    def reason(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        if not self.adaptive_enabled:
            return super().reason(plan, context)

        self.fast_prior.reset()
        anchor = self.base_engine.reason(plan, context)
        beam: list[FlowCandidate | None] = [None]
        all_candidates: list[FlowCandidate] = []
        adaptation_trace: list[Mapping[str, Any]] = []
        compute_trace: list[Mapping[str, Any]] = []
        steps = 0
        total_generated = 0
        total_verifier_calls = 0
        total_expansions = 0

        self._note(
            "adaptive_reason_start",
            max_steps=self.max_steps,
            initial_branches=self.initial_branches,
            max_branches=self.branches,
        )

        for step in range(1, self.max_steps + 1):
            steps = step
            self._note(
                "adaptive_step_start",
                step=step,
                max_steps=self.max_steps,
                parents=len(beam),
                initial_branches=self.initial_branches,
            )
            proposed = self._sample_for_parents(
                plan=plan,
                context=context,
                anchor=anchor,
                beam=beam,
                step=step,
                branches=self.initial_branches,
            )
            total_generated += len(proposed)
            if not proposed:
                break
            proposed.sort(key=lambda item: item.score, reverse=True)
            assessment = self._assessment(proposed, anchor)

            expanded = 0
            if (
                assessment.uncertain
                and self.adaptive_expand
                and self.initial_branches < self.branches
            ):
                extra_per_parent = self.branches - self.initial_branches
                extra = self._sample_for_parents(
                    plan=plan,
                    context=context,
                    anchor=anchor,
                    beam=beam,
                    step=step,
                    branches=extra_per_parent,
                    branch_offset=self.initial_branches,
                    seed_offset=50,
                )
                proposed.extend(extra)
                expanded = len(extra)
                total_generated += expanded
                total_expansions += 1
                proposed.sort(key=lambda item: item.score, reverse=True)
                assessment = self._assessment(proposed, anchor)
                self._note(
                    "adaptive_expand",
                    step=step,
                    added_candidates=expanded,
                    reasons=list(assessment.reasons),
                )

            verifier_calls = 0
            should_verify = (
                self.verifier_enabled
                and self.verify_top_k > 0
                and self.adaptive_verify
                and assessment.uncertain
            )
            if should_verify:
                top = proposed[: self.verify_top_k]
                verified = {
                    candidate.id: self._verify(
                        candidate,
                        anchor,
                        seed=(
                            None
                            if self.seed is None
                            else self.seed + step * 10000 + rank
                        ),
                    )
                    for rank, candidate in enumerate(top)
                }
                verifier_calls = len(verified)
                total_verifier_calls += verifier_calls
                proposed = [verified.get(item.id, item) for item in proposed]
                proposed.sort(key=lambda item: item.score, reverse=True)
                assessment = self._assessment(proposed, anchor)
                self._note(
                    "adaptive_verify",
                    step=step,
                    verifier_calls=verifier_calls,
                    reasons=list(assessment.reasons),
                )

            winner = proposed[0]
            update = self._adapt(winner, anchor)
            if update is not None:
                adaptation_trace.append(
                    {"step": step, "candidate_id": winner.id, **dict(update)}
                )
                proposed = self._rescore_after_adaptation(proposed, anchor)
                assessment = self._assessment(proposed, anchor)
                winner = proposed[0]

            stop = (
                step >= self.min_steps
                and winner.final
                and not winner.tool_requests
                and not assessment.uncertain
            )
            compute_trace.append(
                {
                    "step": step,
                    "parents": len(beam),
                    "initial_branches": self.initial_branches,
                    "candidate_count": len(proposed),
                    "expanded_candidates": expanded,
                    "verifier_calls": verifier_calls,
                    "uncertain": assessment.uncertain,
                    "uncertainty_reasons": list(assessment.reasons),
                    "confidence": assessment.confidence,
                    "evidence_score": assessment.evidence_score,
                    "score_margin": assessment.score_margin,
                    "verifier_score": assessment.verifier_score,
                    "must_continue": assessment.must_continue,
                    "stop": stop,
                }
            )
            self._note(
                "adaptive_step_complete",
                step=step,
                stop=stop,
                uncertain=assessment.uncertain,
                reasons=list(assessment.reasons),
                confidence=assessment.confidence,
                evidence_score=assessment.evidence_score,
                candidate_count=len(proposed),
                verifier_calls=verifier_calls,
            )
            all_candidates.extend(proposed)
            beam = list(proposed[: self.beam_width])
            if stop:
                break

        best = max(all_candidates, key=lambda item: item.score) if all_candidates else None
        solution = self._finalize(anchor, best, all_candidates, steps, adaptation_trace)
        flow = dict(solution.metadata.get("stochastic_flow", {}))
        flow["adaptive_compute"] = {
            "enabled": True,
            "initial_branches": self.initial_branches,
            "max_branches": self.branches,
            "max_steps": self.max_steps,
            "steps_used": steps,
            "generated_candidates": total_generated,
            "verifier_calls": total_verifier_calls,
            "expansions": total_expansions,
            "trace": compute_trace,
        }
        self._note(
            "adaptive_reason_complete",
            steps_used=steps,
            generated_candidates=total_generated,
            verifier_calls=total_verifier_calls,
            expansions=total_expansions,
        )
        return replace(
            solution,
            metadata={**solution.metadata, "stochastic_flow": flow},
        )


__all__ = ["AdaptiveStochasticFlowReasoner", "ComputeAssessment"]
