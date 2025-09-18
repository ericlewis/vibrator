# Vibrator personalization prototype

A production-ready personalization system that matches user behavior with content using slider-based preference modeling. Vibrator uses state-of-the-art instructional embeddings to understand user actions (clicks, writes, reactions) and score content against customizable personality/preference dimensions.

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

## Quickstart

### Installation
```bash
# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
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
scorer = SliderScorer(encoder, sliders)

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

### Chat transcripts

Use `sample_recent_chat_actions` to turn the latest chat turns into `UserAction`s with recency-aware weights. This keeps the transcript lightweight for embedding while aligning with the scoring pipeline.

```python
from datetime import datetime, timedelta, timezone

from vibrator import ChatMessage, sample_recent_chat_actions

now = datetime.now(timezone.utc)
transcript = [
    ChatMessage("We need fun ideas for our study group tonight", now - timedelta(hours=2)),
    ChatMessage("Also looking for ways to keep people accountable", now - timedelta(minutes=30)),
]

actions = sample_recent_chat_actions(
    transcript,
    now=now,
    max_messages=2,
    include_roles=("user",),
    half_life_hours=3.0,
)
```

See `examples/chat_sampling.py` for a full end-to-end walkthrough that scores a new content item using a recent chat transcript.

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

# Batch processing
all_actions = load_batch_actions(user_ids)
all_scores = scorer.score_batch(all_actions, content_items)
```

### Storage options
- **Redis**: Fast embedding cache for real-time scoring
- **PostgreSQL + pgvector**: Persistent vector storage with SQL queries
- **Pinecone/Weaviate**: Managed vector databases for scale

### Monitoring
```python
# Track calibration drift
from vibrator.calibration import calculate_brier_score

actual_engagement = get_user_engagement(predictions)
brier = calculate_brier_score(predictions, actual_engagement)

if brier > threshold:
    recalibrate_model()
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

### Fine-tuning embeddings
```python
# Train custom embeddings on your data
from vibrator.training import fine_tune_encoder

encoder = fine_tune_encoder(
    base_model="instructor-base",
    training_data=your_action_logs,
    loss="triplet"  # or "contrastive"
)
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

## CI/CD

The repository includes GitHub Actions for:
- Running tests on pull requests
- Caching model downloads
- Regression testing against fixtures
- Tracking calibration drift

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
