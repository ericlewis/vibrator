# Data Integration Guide

## Quick Start

Run the integration example to see how to use your own data:
```bash
uv run python examples/integrate_your_data.py
```

## Data Requirements

### 1. User Actions
Your system needs user behavior data with these fields:

**Required:**
- `action_type`: Type of action (click, write, react, read, share, open)
- `content`: Text description of what the user interacted with

**Optional but recommended:**
- `timestamp`: When the action occurred (for recency weighting)
- `user_id`: Which user performed the action
- `weight`: Importance of this action (default: 1.0)

**CSV Format:**
```csv
timestamp,user_id,action_type,content,weight
2024-01-15T10:30:00,user123,click,Article about Python best practices,1.0
2024-01-15T11:00:00,user123,write,Need help with async programming,1.2
```

**JSON Format:**
```json
{
  "actions": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "user_id": "user123",
      "action_type": "click",
      "content": "Article about Python best practices",
      "weight": 1.0
    }
  ]
}
```

### 2. Slider Definitions
Define personality/preference dimensions as text descriptions:

```json
{
  "sliders": {
    "technical": "User prefers detailed technical content",
    "beginner": "User seeks simple explanations",
    "news": "User interested in latest updates"
  }
}
```

### 3. Content to Score
Items to rank/recommend (articles, products, etc.):

```json
{
  "items": [
    "Complete guide to Python async programming",
    "Understanding machine learning theory",
    "Deploy Django app to production"
  ]
}
```

## Integration Steps

### Step 1: Load Your Data

```python
from vibrator import UserAction

# From CSV
actions = load_actions_from_csv('your_data.csv')

# From JSON
actions = load_actions_from_json('your_data.json')

# From Database
actions = query_user_actions(user_id, days_back=7)
```

### Step 2: Define Sliders

```python
slider_definitions = {
    "professional": "Content for work and productivity",
    "personal": "Content for life and relationships",
    "entertainment": "Fun and recreational content"
}
```

### Step 3: Score Content

```python
from vibrator import InstructionalEncoder, SliderScorer

encoder = InstructionalEncoder()
slider_vectors = encode_sliders(encoder, slider_definitions)
scorer = SliderScorer(encoder, slider_vectors)

# Score each content item
for content in your_content:
    scores = scorer.score(actions, content)
    best_match = max(scores, key=lambda x: x.probability)
```

## Action Types

The system understands these action types:

- **write**: User created content (high signal)
- **click**: User selected/opened something
- **react**: User gave feedback (like, upvote)
- **read**: User viewed/consumed content
- **share**: User shared with others
- **open**: User opened/accessed something

Custom action types are supported - the system will adapt.

## Weighting Strategy

Actions can be weighted by:

1. **Type importance**: `write` > `click` > `react` > `read`
2. **Recency**: Recent actions weighted higher
3. **Context**: Work hours vs personal time
4. **Custom rules**: Your domain-specific logic

Example with recency decay:
```python
age_hours = (now - action_time).total_seconds() / 3600
weight = base_weight * (0.95 ** (age_hours / 24))  # 5% decay per day
```

## Production Considerations

### Performance
- Cache encoder embeddings (they're expensive to compute)
- Batch encode multiple items at once
- Consider using Redis or pgvector for embedding storage

### Scale
- Process actions in batches for multiple users
- Use async processing for real-time updates
- Implement sliding window for action history

### Quality
- Monitor calibration scores over time
- A/B test different slider definitions
- Log which sliders users engage with

## Example Data Files

Create these files to test with your data:

**data/user_actions.csv:**
```csv
timestamp,user_id,action_type,content,weight
2024-01-20T09:00:00,user1,write,Looking for Python debugging tips,1.5
2024-01-20T09:30:00,user1,click,Advanced Python patterns article,1.0
2024-01-20T10:00:00,user1,react,Liked async/await tutorial,0.8
```

**data/sliders.json:**
```json
{
  "sliders": {
    "technical": "Detailed technical documentation",
    "tutorial": "Step-by-step learning guides",
    "news": "Industry news and updates"
  }
}
```

**data/content.json:**
```json
{
  "items": [
    "Python 3.12 new features explained",
    "How to debug async code effectively",
    "Building your first REST API"
  ]
}
```

Then run:
```bash
uv run python examples/integrate_your_data.py
```