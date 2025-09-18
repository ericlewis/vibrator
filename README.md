# Vibrator personalization prototype

This repository contains an MVP for slider-style personalization that aligns user behavior, candidate content, and slider prototypes in a common embedding space. It uses instruction-tuned sentence embeddings, lightweight action-aware prompts, and optional calibration hooks so you can iterate on weighting strategies quickly.

## Highlights
- **Instructional embeddings**: wraps `hkunlp/instructor-base` so items and heterogeneous actions are encoded with controllable prompts.
- **Action aggregation**: converts clicks, writes, reactions, and reads into a shared latent representation with per-action weighting and recency decay.
- **Synthetic bootstrapping**: bundled generators create placeholder sliders, actions, and content so you can demo without production data.
- **Contextual weighting**: exposes helpers to adjust slider weights by user segment or temporal context (e.g., work hours).
- **Calibration ready**: ships temperature scaling and isotonic regression utilities to keep slider probabilities well-calibrated.
- **Temporal adaptation friendly**: inspired by [VIBE: Topic-Driven Temporal Adaptation for Twitter Classification](https://arxiv.org/abs/2310.10191), the design keeps hooks for future online updates as language drifts.

## Quickstart
1. Create a virtual environment and install dependencies with uv:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```
2. Run the end-to-end example (downloads the Instructor model on first run):
   ```bash
   uv run python examples/mvp.py
   ```
   The script prints raw scores, calibrated probabilities, and feature contributions for three illustrative sliders.
3. Execute the test suite (first run downloads the Instructor model):
   ```bash
   uv run pytest
   ```
4. Explore the synthetic demo:
   ```bash
   uv run python examples/synthetic_demo.py
   ```

## Key modules
- `vibrator/actions.py` – data model plus prompt prefixes for heterogeneous actions.
- `vibrator/encoder.py` – `InstructionalEncoder` wrapper around `SentenceTransformer` with layer/L2 normalization.
- `vibrator/profile.py` – utilities to combine action embeddings into a user profile.
- `vibrator/pipeline.py` – `SliderScorer` that blends user, item, and slider vectors and returns calibrated probabilities.
- `vibrator/calibration.py` – temperature scaling and isotonic regression helpers for probability calibration.
- `vibrator/weights.py` – contextual weight adjustments for segments and time-of-day rules.
- `vibrator/synthetic.py` – generates synthetic sliders, items, and action logs for smoke tests.
- `fixtures/` – holds regression fixtures with expected probabilities for drift detection.

## Suggested extensions
1. **Fine-tune embeddings** – with behavior logs, add a contrastive head (triplet loss) to encourage thread-level cohesion, using this repo as the inference scaffold.
2. **Personalized weighting** – learn per-segment weight overrides or train a gating MLP that emits feature weights from context features.
3. **Temporal drift adaptation** – follow VIBE-style topic bottlenecks or streaming retrieval to refresh slider vectors without full retraining.
4. **Calibration monitoring** – log Brier scores for each slider and automate temperature re-fitting when drift is detected.
5. **Cold-start defaults** – layer in cohort priors so new users bootstrap from population averages until enough actions accrue.

## Repository status
This is a prototype to accelerate experimentation. Expect to wire it into your serving stack, add persistence for embeddings, and integrate evaluation pipelines (offline and A/B) as you iterate.

## CI and regression fixtures
- GitHub Actions workflow `.github/workflows/tests.yml` caches `~/.cache/torch/sentence_transformers` and `~/.cache/huggingface` so repeated runs reuse the Instructor model.
- Drop updated anonymized logs into `fixtures/regression_logs.json` and refresh `expected_probabilities` by running `uv run python scripts/update_regression_fixture.py` (or rerun `tests/test_regression_fixture.py`) to capture new baselines.
- Use `pytest -m 'not integration'` to skip model-dependent tests when iterating quickly.
