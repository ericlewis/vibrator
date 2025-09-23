# Vibrator

A personalization system that matches user behavior with content using slider-based preference modeling. Vibrator uses state-of-the-art instructional embeddings to understand user actions (clicks, writes, reactions) and score content against customizable personality/preference dimensions.

## What it does

Vibrator helps you build recommendation systems that:
- Learn from heterogeneous user actions (clicks, writes, reactions, reads, shares)
- Score content against multiple personality dimensions (e.g., technical vs beginner-friendly)
- Adapt to different contexts (work hours, user segments, device types)
- Provide calibrated probability scores for A/B testing and decision-making

## Core capabilities
- **Instructional embeddings**: Uses `hkunlp/instructor-base` to encode user actions and content with context-aware prompts
- **Multi-action learning**: Intelligently weights different action types (writes > clicks > reactions > reads) with configurable importance
- **Real-time personalization**: Processes user behavior streams to update preferences dynamically
- **Calibrated scoring**: Returns well-calibrated probabilities using temperature scaling and isotonic regression
- **Context awareness**: Adjusts recommendations based on time-of-day, user segments, and custom rules
- **Production ready**: Includes synthetic data generation, regression testing, and CI/CD setup

## What is a slider?

A slider is a named preference dimension described in natural language (a short prompt) that defines what “match” means along that axis.

How it works at a glance:
- Encoder embeds the slider text, the user’s profile (from actions), and the candidate item in the same space.
- The scorer combines four features — user↔slider, item↔slider, user↔item, and recency — then calibrates to a per‑slider probability. See “How to interpret the score”.

Anatomy of a slider:
- **name**: stable identifier (e.g., `technical`, `beginner`, `creative`)
- **description**: 1–12 words capturing intent (imperative, unambiguous)
- optional **calibration/weights**: per‑slider tuning for trustworthy scores

Examples:
- `technical`: "Detailed technical documentation"
- `beginner`: "Simple, step‑by‑step explanations"
- `creative`: "Imaginative or unconventional content"

Tips:
- Start with 3–7 sliders; keep descriptions short and concrete.
- Avoid overlapping meanings; split axes if needed (e.g., tone vs depth).
- Cache slider embeddings for speed.

### How to design good sliders

- Use single‑intent phrasing; avoid conjunctions ("and", "or").
- Keep descriptions concise (4–12 words) and action‑oriented.
- Prefer orthogonal axes; minimize overlap between sliders.
- Name sliders in your product’s language (user‑facing, stable identifiers).
- Validate with small labeled sets (5–10 examples per slider) and iterate.
- Monitor per‑slider calibration; add temperature/isotonic calibrators if needed.
- Version sliders when meanings change; avoid silent redefinitions.

### Anti-patterns

- Vague or multi-intent descriptions (e.g., "technical and creative").
- Overlapping sliders that measure the same concept under different names.
- Hidden context or internal jargon the encoder cannot infer.
- Long, paragraph-like descriptions; exceed ~12 words.
- Redefining slider meanings without versioning and recalibration.
- Sliders tied to sensitive or policy-violating attributes.

### Examples: good vs bad

Good (concise, single-intent, orthogonal):
```json
{
  "sliders": {
    "technical": "Detailed technical documentation",
    "beginner": "Simple, step-by-step explanations",
    "creative": "Imaginative or unconventional content"
  }
}
```

Bad (avoid these patterns):
- "technical and creative" — multi-intent
- "good content" — vague
- "for power users at work on mobile" — hidden context
- Very long paragraph — exceeds ~12 words, hard to embed consistently

## Quickstart

### Installation
```bash
# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Install the local package so `import vibrator` works
uv pip install -e .
```

If you prefer not to install the package, you can run examples by temporarily adding the project root to PYTHONPATH:
```bash
PYTHONPATH=. uv run python examples/mvp.py
```

### Try the examples

1. **Basic usage** - See how sliders work with simple actions:
   ```bash
   uv run python examples/mvp.py
   ```

2. **Synthetic data** - Test with auto-generated content:
   ```bash
   uv run python examples/synthetic_demo.py
   ```

3. **Your own data** - Integrate with your user actions:
   ```bash
   uv run python examples/integrate_your_data.py
   ```

### Run tests
```bash
uv run pytest  # Downloads model on first run
```

### First run notes
- On first use, Sentence Transformers will download the model weights (hkunlp/instructor-base). This requires an internet connection and roughly ~1GB of free disk space.
- The examples use `uv` to run inside the local virtual environment created by `uv venv`.

### Troubleshooting
- ModuleNotFoundError: No module named 'vibrator'
  - Fix: Install the package in editable mode: `uv pip install -e .`
  - Alternative: Run with the project root on PYTHONPATH: `PYTHONPATH=. uv run python examples/mvp.py`
- Verify the install
  ```bash
  uv run python -c "import vibrator; print(vibrator.__version__ if hasattr(vibrator, '__version__') else 'ok')"
  ```

### Development setup
- Packaging: This project uses `pyproject.toml` with the Hatchling backend to package the local `vibrator` module.
- Editable install for local changes:
  ```bash
  uv pip install -e .
  ```
- Keep dependencies synced in `requirements.txt`; install with:
  ```bash
  uv pip install -r requirements.txt
  ```

## Using your own data

### Quick integration
```python
from vibrator import InstructionalEncoder, SliderScorer, UserAction

# 1. Define your personality/preference dimensions
sliders = {
    "technical": "User prefers technical documentation",
    "beginner": "User seeks simple tutorials",
    "creative": "User enjoys creative content"
}

# 2. Load user actions from your system
actions = [
    UserAction("write", "How do I optimize database queries?", weight=1.5),
    UserAction("click", "PostgreSQL indexing guide", weight=1.0),
    UserAction("react", "Upvoted SQL performance tips", weight=0.8)
]

# 3. Score content
encoder = InstructionalEncoder()
slider_vectors = {name: encoder.encode_items([desc])[0] for name, desc in sliders.items()}
scorer = SliderScorer(encoder, slider_vectors)

content = "Advanced PostgreSQL query optimization techniques"
scores = scorer.score(actions, content)

# Best matching slider
best = max(scores.items(), key=lambda x: x[1].probability)
print(f"Recommended as: {best[0]} (confidence: {best[1].probability:.2%})")
```

### How to interpret the score

- `SliderScorer.score` returns a `SliderOutput` for every slider with the `raw_score`, `probability`, contributing feature values, and the weights that produced the result.
- `raw_score` is a weighted combination of four cosine-similarity style features (user↔slider, item↔slider, user↔item, recency). Because embeddings are L2-normalized, it behaves like a logit capturing how strongly the item matches that slider given the user's history.
- `probability` converts that raw logit into a calibrated confidence. If you provide per-slider calibrators (temperature or isotonic) the value aligns with observed engagement rates; otherwise the scorer applies a sigmoid.
- Use `probability` for ranking, gating, and thresholds in matchmaking or other downstream decisions. Surface the feature breakdown when you need explainability or to bias business logic via the `context` argument (segments, work hours, boosts) before scoring.

### Usage ideas

- **Chat transcripts** – Use `sample_recent_chat_actions` to roll the latest chat turns into weighted `UserAction`s so you can score content with recency baked in (`examples/chat_sampling.py`).
- **Chat room matchmaking** – Match members to rooms by scoring each room’s pitch against sampled chats, plus context-sensitive feature overrides that reward fresh tone matches (`examples/chat_room_matchmaking.py`).
- **Moderation penalties** – After scoring, scan actions for disallowed phrases and subtract a hefty penalty to bury the probability (e.g., the `@tawny` “corn dog” guardrail in `examples/chat_room_penalty.py`).
- **Recipe personalization** – Blend slider probabilities into a composite to pick the best dinner tweak based on spice tolerance, macro goals, and prep constraints (`examples/recipe_personalization.py`).
- **Cosmic weirdness** – Stress-test the pipeline with surreal sliders (void whispers, lava surfing) and cursed phrase penalties in `examples/weird_exotic_demo.py`.
- **Day planning vibe meter** – Score possible calendar blocks against sliders like `deep_focus`, `social_energy`, or `recharge`, using your latest reflections as actions to choose an agenda that fits today’s aura.
- **Aura profile smoothing** – Persist per-slider probabilities as a “vibe vector,” update it daily with new actions, and expose it across matchmaking, status badges, or moderation triage.
- **Confidence watchdog** – Log predictions + outcomes, fit calibrators via `TemperatureCalibrator`/`IsotonicCalibrator`, and monitor Brier scores to quantify how trustworthy your probabilities are.
- **Fusion layers** – Combine multi-modal signals (chat + purchase + sensor data) by turning each into `UserAction`s with custom weights, then score interactive experiences that blend commerce, community, and gamified quests.
- **Counterfactual testing** – Clone a user’s action set, tweak weights (e.g., boost `write` actions or remove toxic phrases), rerun scoring to see how matches shift before making policy changes.
- **Self-adapting loop** – Log slider scores, contexts, and outcomes; auto-fit calibrators; mine fresh signals for new sliders or weight overrides; and guard the loop with metrics (Brier score, drift detectors, rollout gates) so the system keeps evolving safely.

### Data formats

#### User actions (CSV)
```csv
timestamp,user_id,action_type,content,weight
2024-01-15T10:30:00,user123,click,Python async tutorial,1.0
2024-01-15T11:00:00,user123,write,Need help with decorators,1.5
```

#### User actions (JSON)
```json
{
  "actions": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "user_id": "user123",
      "action_type": "click",
      "content": "Python async tutorial",
      "weight": 1.0
    }
  ]
}
```

#### Slider definitions
```json
{
  "sliders": {
    "technical": "Detailed technical content",
    "beginner": "Simple explanations for newcomers",
    "practical": "Hands-on tutorials and examples"
  }
}
```

See [`DATA_INTEGRATION.md`](DATA_INTEGRATION.md) for complete integration guide.

## How it works

### 1. Action encoding
User actions are converted to embeddings using instructional prompts:
- "User wrote: How to implement OAuth?" → High-signal intent
- "User clicked on: Authentication guide" → Medium-signal interest
- "User reacted to: Security best practices" → Low-signal preference

### 2. Profile building
Actions are aggregated with:
- **Type weighting**: writes (1.5x) > clicks (1.0x) > reactions (0.8x)
- **Recency decay**: Recent actions weighted higher
- **Context adjustment**: Work hours vs personal time

### 3. Slider scoring
Content is scored against each slider dimension:
1. Encode content and sliders in the same space
2. Compute similarity between user profile and content
3. Apply calibration for accurate probabilities
4. Return scores with feature explanations

## Architecture

```
User Actions → Instructional Encoder → Action Embeddings
                                            ↓
                                     Profile Aggregation
                                            ↓
Content Text → Instructional Encoder → Content Embedding
                                            ↓
                                      Slider Scoring
                                            ↓
                                   Calibrated Probabilities
```

## Key modules
- `vibrator/actions.py` – User action data model with type-specific prompts
- `vibrator/encoder.py` – Instructional embedding wrapper with normalization
- `vibrator/profile.py` – Aggregates actions into user preference vectors
- `vibrator/pipeline.py` – Main scorer combining user, content, and slider signals
- `vibrator/calibration.py` – Probability calibration for accurate confidence scores
- `vibrator/weights.py` – Context-aware weight adjustments
- `vibrator/synthetic.py` – Generate test data for development
- `examples/integrate_your_data.py` – Complete integration example

## Production deployment

### Performance optimization
```python
# Cache embeddings
slider_vectors = encoder.encode_items(slider_texts)
cache.set("sliders", slider_vectors, ttl=3600)

# Batch processing (example)
all_actions = load_batch_actions(user_ids)
all_scores = [scorer.score(actions, item) for actions, item in zip(all_actions, content_items)]
```

### Storage options
- **Redis**: Fast embedding cache for real-time scoring
- **PostgreSQL + pgvector**: Persistent vector storage with SQL queries
- **Pinecone/Weaviate**: Managed vector databases for scale

### Monitoring
```python
# Track calibration drift (Brier score)
import numpy as np

# predictions: List[float] in [0,1], actual: List[int] 0/1
actual = get_user_engagement(predictions)
p = np.asarray(predictions, dtype=np.float32)
y = np.asarray(actual, dtype=np.float32)
brier = float(np.mean((p - y) ** 2))

if brier > threshold:
    # Refit calibrators (TemperatureCalibrator / IsotonicCalibrator) or revisit features
    pass
```

## Advanced usage

### Custom action types
```python
# Add domain-specific actions
UserAction("purchase", "Premium Python course", weight=2.0)
UserAction("bookmark", "API reference guide", weight=0.9)
UserAction("search", "async await examples", weight=0.7)
```

### Dynamic weighting
```python
# Adjust weights by context
context = {
    "segment": "power_user",
    "is_work_hours": True,
    "device": "mobile"
}

scores = scorer.score(actions, content, context=context)
```



## Testing

```bash
# All tests
uv run pytest

# Fast iteration (no model download)
uv run pytest -m "not integration"

# Specific module
uv run pytest tests/test_pipeline.py

# With coverage
uv run pytest --cov=vibrator
```

## CI/CD (suggested)

Recommended tasks:
- Run tests on pull requests
- Cache model downloads
- Regression-test against fixtures
- Track calibration metrics

## Next steps

1. **Quick start**: Run `examples/integrate_your_data.py` with your data
2. **Customize sliders**: Define dimensions relevant to your domain
3. **Add storage**: Implement Redis/PostgreSQL for production scale
4. **Deploy API**: Wrap scorer in FastAPI/Flask endpoint
5. **Monitor**: Track calibration scores and user engagement
6. **Iterate**: A/B test different slider definitions and weights

## References

- [VIBE: Topic-Driven Temporal Adaptation](https://arxiv.org/abs/2310.10191) - Inspiration for temporal adaptation design
- [Instructor embeddings](https://instructor-embedding.github.io/) - Foundation model for encoding
- [Calibration methods](https://scikit-learn.org/stable/modules/calibration.html) - Probability calibration techniques

## License

Licensed under the MIT License. See `LICENSE` for details.