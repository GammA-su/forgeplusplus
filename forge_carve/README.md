# FORGE++-CARVE MVP

A minimal, buildable MVP repo for the FORGE++ (primal-dual latent proof optimizer) + CARVE (metamorphic verifier mesh) system.

This project is offline-only, deterministic, and uses synthetic data generators.

See scripts under `forge_carve/scripts/` for the full data -> train -> eval pipeline.

## Quickstart (uv)

```bash
uv sync
uv run pytest -q
uv run python scripts/make_data_schema.py --seed 123
uv run python scripts/make_data_math.py --seed 123
uv run python scripts/make_data_csp.py --seed 123
uv run python scripts/train_mvp.py --config configs/train_mvp.yaml
uv run python scripts/eval_suite.py --config configs/eval_suite.yaml
```
