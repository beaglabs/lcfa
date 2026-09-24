# LCFA stochastic-flow Mac preset

This preset is sized for Apple Silicon systems with limited unified memory. It uses:

- `mlx-local` backbone
- `mlx-fast` session-scoped parametric prior
- `mlx-community/Qwen3-4B-Instruct-2507-4bit`
- 2 branches / beam width 2 / up to 3 reasoning steps
- 4K active model context

The model is intentionally not committed to git. Download it into the repo-local ignored `models/` directory:

```bash
python -m pip install -e '.[mac]'
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --local-dir models/Qwen3-4B-Instruct-2507-4bit
```

Smoke-test MLX directly:

```bash
mlx_lm.generate \
  --model models/Qwen3-4B-Instruct-2507-4bit \
  --prompt 'Return one sentence saying LCFA is ready.' \
  --max-tokens 32
```

Then run LCFA:

```bash
lcfa-bench run builtin:all \
  --artifact artifacts/lcfa-stochastic-flow-mac \
  --prior-snapshot /tmp/lcfa-prior.safetensors \
  --repeats 1 \
  -o /tmp/lcfa-mac.json
```

The artifact's backbone path is `../../models/Qwen3-4B-Instruct-2507-4bit`, resolved relative to this artifact directory. Override it at runtime with `--backbone-path /absolute/path/to/model` if desired.

The first run can take longer because the model must be downloaded and loaded. The canonical 17 cases are regression tests; they are not a SOTA reasoning-quality benchmark.
