# Frozen LCFA evaluation suites

These suites are evaluation targets, not training corpora. Keep their membership stable across model/runtime revisions.

- `gsm8k-system-240.json`: 240 fixed GSM8K **test** row indices. Intended for cheap reasoning/system regression.
- `swebench-lite.json`: the complete official SWE-bench Lite evaluation set. Intended for milestone software-engineering evaluation.
- `ale-lcfa-48.txt`: 48 fixed unlicensed Agents' Last Exam tasks spanning the public professional-task domains. Intended for long-horizon agent evaluation.

Machine-readable benchmark reports are preserved as JSON. GitHub Actions renders the same reports as a Markdown Job Summary using `lcfa-bench-card` and `$GITHUB_STEP_SUMMARY`.
