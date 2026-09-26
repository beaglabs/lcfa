"""LCFA public API."""

from .actions import RecommendationActionCompiler
from .adaptation import FEATURE_NAMES, FastPrior, MLXFastPrior, NullFastPrior, NumpyFastPrior, create_fast_prior
from .artifact import ARTIFACT_FORMAT, STOCHASTIC_FLOW_ARCHITECTURE, WEIGHTED_OUTPUT_ARCHITECTURE, ArtifactError, ArtifactManifest, ArtifactReasoner, ArtifactWeights, SafetensorsReasonerAdapter, artifact_architectures, load_artifact_manifest, load_artifact_reasoner, register_artifact_architecture
from .backbones import BackboneSample, LlamaCppBackbone, MLXCausalBackbone, ReferenceBackbone, StochasticBackbone, TransformersCausalBackbone, create_backbone
from .bench import BENCH_FORMAT, COMPARISON_FORMAT, REPORT_FORMAT, AssertionResult, BenchmarkCase, BenchmarkError, BenchmarkReport, BenchmarkRunner, BenchmarkSubject, BenchmarkSuite, BenchmarkSummary, CaseResult, ComparisonReport, ExpectedValue, ReasonerSubject, benchmark_report_from_dict, benchmark_report_to_dict, benchmark_suite_from_dict, benchmark_suite_to_dict, compare_reports, dump_benchmark_report, dump_benchmark_suite, dumps_benchmark_report, dumps_benchmark_suite, dumps_comparison, load_benchmark_report, load_benchmark_suite, loads_benchmark_report, loads_benchmark_suite
from .bench_corpus import all_suites, suite_by_id
from .cognitive import COGNITIVE_STATE_FORMAT, CognitiveState, Hypothesis, SemanticInvestigator
from .engine import LCFA
from .operator_lib import ConstraintViolation, register_reasoning_operators
from .operators import register_core_operators
from .profile import Profile, ProfileError, load_profile
from .protocol import ActionGraph, ActionNode, ActionResult, ActionRun, EntityRef, EvidenceRef, EvidenceValue, ExecutionContext, Finding, OperatorResult, PlanNode, ReasoningPlan, Recommendation, SolutionState
from .recurrent_transitions import ACTION_VOCAB, RECURRENT_TRANSITION_FORMAT, RecurrentTransition, dump_transitions, episode_to_transitions, load_episode, load_transitions, prepare_transition_file
from .registry import ActionRegistry, ActionSpec, OperatorRegistry, OperatorSpec
from .repo_index import IndexResult, PythonRepoIndexer
from .runtime import ActionExecutor, ActionPolicyError, PlanError, ReasoningExecutor
from .rwkv_controller import DEFAULT_RWKV_MODEL, RWKV_CONTROLLER_FORMAT, RWKVControllerError, RWKVRecurrentPolicy, RecurrentDecision, RecurrentPolicy
from .rwkv_semantic import RWKVSemanticBackbone, load_rwkv_semantic_backbone
from .semantic_agent import SEMANTIC_AGENT_TRAJECTORY_FORMAT, SemanticAgentEpisode, SemanticAgentStep, SemanticWorkspaceAgent
from .semantic_graph import SEMANTIC_GRAPH_FORMAT, ConceptEdge, ConceptNode, ConceptSnapshot, ContentRef, SemanticGraph, SQLiteSemanticGraph
from .serde import IR_FORMAT, SerializationError, dump_ir, dumps_ir, from_ir_dict, load_ir, loads_ir, to_ir_dict
from .state import MemoryStateStore, StateConflictError, StateIdentity, StateNotFoundError, StateSnapshot, StateStore, StateStoreError, canonical_bytes, content_hash
from .stochastic import FlowCandidate, FlowToolRequest, StochasticFlowReasoner
from .workspace_actions import WorkspaceActionError, compile_cognitive_actions, register_workspace_actions
from .zplug import LATENT_PACKET_FORMAT, ZPLUG_FORMAT, HashTextZPlug, LatentDelta, LatentPacket, MLXTextZPlug, Observation, OutputRequest, ZContext, ZPlug, ZPlugError, ZPlugManifest, ZPlugRegistry, load_zplug_manifest, zplug_manifest_from_dict, zplug_manifest_to_dict
from .latent_flow import LATENT_FLOW_ARCHITECTURE, LATENT_STATE_FORMAT, LatentFlowReasoner, LatentState, NumpyLatentCore
from . import latent_artifact as _latent_artifact

# Keep mixed-benchmark helper bindings corrected for both CLI and direct imports.
from . import bench_mixed_fixups as _bench_mixed_fixups

__all__ = [name for name in globals() if not name.startswith("_")]
