# LCFA zplugs and `lcfa.latent-flow.v1`

LCFA's latent-first runtime treats modality support as an extension boundary rather than a core-mode switch.

## Contracts

- `lcfa.zplug.v1` describes an extension that can encode/enrich/decode one or more modalities.
- `lcfa.latent-packet.v1` carries plug-native features plus provenance/evidence and optional shared/private latent values.
- `lcfa.latent-state.v1` is the modality-neutral state evolved by the latent core.
- `lcfa.latent-flow.v1` is the first artifact architecture that learns and executes latent cognition while preserving the normal `ReasoningPlan -> SolutionState` boundary.

A **zplug** is deliberately narrower than a generic plugin: it is an adapter that plugs a capability into LCFA's latent state `z`.

```text
text/image/audio/sensor/... observation
                 |
                 v
              zplug
                 |
          LatentPacket
                 |
                 v
        LCFA latent encoder
                 |
                 z0
                 |
          latent dynamics
        z0 -> z1 -> ... -> zn
                 |
          typed SolutionState
```

## Text-first implementation

The first real zplug is `MLXTextZPlug`. It loads a local `mlx-lm` model and extracts the final hidden state **before the LM head**. No text generation is required for enrichment. Qwen therefore supplies semantic features while the LCFA encoder and dynamics learn their own latent representation.

`HashTextZPlug` is a deterministic reference fixture for CI only.

## Training objective

The initial trainer implements masked latent prediction with an EMA target encoder:

```text
masked text ---------------------> frozen text zplug ---> student features
full text -----------------------> frozen text zplug ---> teacher features
                                                               |
student features -> LCFA encoder -> predictor -----------------+--> latent loss
                                      ^                        |
                                      |                        v
                               trainable dynamics       EMA target encoder
```

The frozen backbone is not optimized. The trainable state consists of:

- LCFA student projection/encoder
- LCFA latent predictor/dynamics
- EMA LCFA target encoder (non-gradient target)

A variance regularizer is included to discourage representation collapse.

The first runtime does **not** activate a learned `SolutionState` decoder. It attaches the learned latent state and metadata to the grounded LCFA-Zero solution. A solution decoder should only be enabled after external reasoning benchmarks show that the learned latent path improves correctness under explicit gates.

## Prepare a text dataset

Input is a UTF-8 file with one document/example per line.

```bash
lcfa-latent prepare corpus.txt \
  --mask-ratio 0.25 \
  --seed 7 \
  -o data/text-latent.jsonl
```

Each row contains `student_text` (masked view) and `teacher_text` (full view).

## Cache frozen Qwen features

On Apple Silicon, the local Qwen model can be used as the first text zplug:

```bash
lcfa-latent encode data/text-latent.jsonl \
  --model models/Qwen3-4B-Instruct-2507-4bit \
  --max-tokens 2048 \
  -o data/qwen3-4b-latents.safetensors
```

This step is intentionally separable from training. Once the feature cache exists, Qwen no longer needs to remain resident during LCFA latent training.

## Train an LCFA latent artifact

```bash
lcfa-latent train data/qwen3-4b-latents.safetensors \
  --output artifacts/lcfa-latent-text-v0 \
  --zplug-model-path ../../models/Qwen3-4B-Instruct-2507-4bit \
  --latent-dim 256 \
  --hidden-dim 512 \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 0.001
```

The output directory is a normal LCFA artifact:

```text
artifacts/lcfa-latent-text-v0/
├── artifact.json
└── model.safetensors
```

The safetensors contain only the LCFA latent model, not the Qwen weights.

## Benchmark the trained representation path

```bash
lcfa-bench run builtin:all \
  --artifact artifacts/lcfa-latent-text-v0 \
  --repeats 1 \
  -o /tmp/lcfa-latent.json
```

At this stage benchmark correctness remains grounded by LCFA-Zero. The report metadata exposes the latent state id/dimension and artifact identity. The next research slice should add benchmark-specific latent heads and compare:

1. frozen Qwen alone
2. Qwen + stochastic-flow
3. frozen Qwen enrichment + trained LCFA latent-flow

## Multimodal extension

Future modalities implement the same zplug boundary rather than creating new LCFA modes:

```text
vision.zplug -> LatentPacket --\
audio.zplug  -> LatentPacket ----> LCFA shared latent state
lidar.zplug  -> LatentPacket --/
```

Each zplug may retain private modality state while projecting useful abstractions into a shared latent representation. The core latent dynamics and governance contracts do not need to change when a new modality is installed.
