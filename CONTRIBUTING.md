# Contributing

- `main` is always runnable. Never push directly to `main`.
- Branch names: `feature/<stage-or-topic>-<name>` e.g. `feature/stage05-transaction-ananya`.
- Open a pull request; at least one teammate reviews before merging.
- Keep commits small, with clear messages.
- Never commit datasets, API keys or `.env` files (see `.gitignore`).
- Every new module gets at least one test in `tests/`.
- Record decisions in `docs/decision_log.md` (what we chose, why, date).
- Dataset claims ("real", "synthetic", label source) must be recorded in `data/datasets.yaml`.
