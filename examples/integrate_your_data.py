"""Example showing how to integrate your own data with the Vibrator system."""
from __future__ import annotations

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from vibrator import InstructionalEncoder, SliderScorer, UserAction
from vibrator.utils import stable_now


def load_actions_from_csv(csv_path: str) -> List[UserAction]:
    """Load user actions from a CSV file.

    Expected CSV format:
    timestamp,user_id,action_type,content,weight
    2024-01-15T10:30:00,user123,click,Article about Python best practices,1.0
    2024-01-15T11:00:00,user123,write,Need help with async programming,1.2
    2024-01-15T11:30:00,user123,react,Upvoted debugging tips,0.8
    """
    actions = []
    # Use a stable reference time for reproducibility; override with VIBRATOR_NOW_ISO
    now = stable_now()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Calculate age in hours for recency decay
            ts = datetime.fromisoformat(row['timestamp'])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.now().astimezone().tzinfo)  # assume local if naive
            # Convert all to UTC-aware for subtraction
            timestamp = ts.astimezone(tz=now.tzinfo)
            age_hours = (now - timestamp).total_seconds() / 3600

            action = UserAction(
                action_type=row['action_type'],
                content=row['content'],
                weight=float(row.get('weight', 1.0)),
                metadata={
                    'user_id': row['user_id'],
                    'age_hours': str(age_hours),
                    'timestamp': row['timestamp']
                }
            )
            actions.append(action)

    return actions


def load_actions_from_json(json_path: str) -> List[UserAction]:
    """Load user actions from a JSON file.

    Expected JSON format:
    {
        "actions": [
            {
                "timestamp": "2024-01-15T10:30:00",
                "user_id": "user123",
                "action_type": "click",
                "content": "Article about Python best practices",
                "weight": 1.0,
                "metadata": {"source": "homepage", "device": "mobile"}
            }
        ]
    }
    """
    actions = []
    # Use a stable reference time for reproducibility; override with VIBRATOR_NOW_ISO
    now = stable_now()

    with open(json_path, 'r') as f:
        data = json.load(f)

    for item in data['actions']:
        # Calculate age in hours for recency decay
        ts = datetime.fromisoformat(item['timestamp'])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.now().astimezone().tzinfo)  # assume local if naive
        timestamp = ts.astimezone(tz=now.tzinfo)
        age_hours = (now - timestamp).total_seconds() / 3600

        # Merge provided metadata with calculated fields
        metadata = item.get('metadata', {})
        metadata.update({
            'user_id': item['user_id'],
            'age_hours': str(age_hours),
            'timestamp': item['timestamp']
        })

        action = UserAction(
            action_type=item['action_type'],
            content=item['content'],
            weight=float(item.get('weight', 1.0)),
            metadata=metadata
        )
        actions.append(action)

    return actions


def load_actions_from_database() -> List[UserAction]:
    """Example of loading from a database.

    Replace this with your actual database connection and query.
    """
    # Example with SQLAlchemy (you'd need to install it):
    # from sqlalchemy import create_engine
    # engine = create_engine('postgresql://user:pass@localhost/db')
    #
    # query = '''
    #     SELECT timestamp, user_id, action_type, content, weight
    #     FROM user_actions
    #     WHERE user_id = %s
    #     AND timestamp > NOW() - INTERVAL '7 days'
    #     ORDER BY timestamp DESC
    # '''
    #
    # with engine.connect() as conn:
    #     result = conn.execute(query, user_id)
    #     actions = []
    #     for row in result:
    #         action = UserAction(...)
    #         actions.append(action)

    # For now, return placeholder
    return [
        UserAction("click", "Database-loaded content", weight=1.0)
    ]


def load_slider_definitions(config_path: str) -> Dict[str, str]:
    """Load your custom slider definitions.

    Expected format:
    {
        "sliders": {
            "technical": "User prefers detailed technical content and documentation",
            "beginner_friendly": "User seeks simple explanations and tutorials",
            "news": "User interested in latest industry news and updates",
            "practical": "User wants hands-on examples and actionable advice"
        }
    }
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['sliders']


def load_content_to_score(source: str) -> List[str]:
    """Load content items to score against user profile.

    This could be articles, products, recommendations, etc.
    """
    # Example: Load from JSON file
    if source.endswith('.json'):
        with open(source, 'r') as f:
            data = json.load(f)
            return data.get('items', [])

    # Example: Load from text file (one item per line)
    elif source.endswith('.txt'):
        with open(source, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    # Add other sources as needed
    return []


def main():
    """Complete example of integrating your data."""

    # Step 1: Initialize the encoder (downloads model on first run)
    print("Initializing encoder...")
    encoder = InstructionalEncoder()

    # Step 2: Load and encode your slider definitions
    # Replace with your actual slider config
    slider_definitions = {
        "technical": "User prefers detailed technical content with code examples",
        "conceptual": "User seeks high-level explanations and theory",
        "practical": "User wants actionable tutorials and how-to guides",
    }

    # Or load from file:
    # slider_definitions = load_slider_definitions('config/sliders.json')

    print(f"Encoding {len(slider_definitions)} slider definitions...")
    slider_vectors = dict(
        zip(slider_definitions.keys(), encoder.encode_items(slider_definitions.values()))
    )

    # Step 3: Load user actions from your data source
    # Choose one of these approaches:

    # Option A: From CSV
    # actions = load_actions_from_csv('data/user_actions.csv')

    # Option B: From JSON
    # actions = load_actions_from_json('data/user_actions.json')

    # Option C: From database
    # actions = load_actions_from_database()

    # For demo, use example actions
    actions = [
        UserAction("write", "How do I implement async/await in Python?", weight=1.5),
        UserAction("click", "Tutorial: Building REST APIs with FastAPI", weight=1.0),
        UserAction("react", "Upvoted explanation of Python decorators", weight=0.8),
        UserAction("read", "Python 3.12 release notes", weight=0.6),
    ]

    print(f"Loaded {len(actions)} user actions")

    # Step 4: Initialize the scorer
    scorer = SliderScorer(encoder, slider_vectors)

    # Step 5: Score content items
    # Replace with your actual content
    content_items = [
        "Complete guide to Python async programming with real examples",
        "Understanding the theory behind machine learning algorithms",
        "Step-by-step: Deploy a Django app to production",
    ]

    # Or load from file:
    # content_items = load_content_to_score('data/articles.json')

    # Add context about the user/session
    context = {
        "segment": "developer",  # User segment
        "is_work_hours": True,    # Time-based context
        "platform": "web",        # Device/platform
    }

    print("\n" + "="*60)
    print("SCORING RESULTS")
    print("="*60)

    for item_text in content_items:
        print(f"\nContent: {item_text[:80]}...")
        scores = scorer.score(actions, item_text, context=context)

        # Find best matching slider
        best_slider = max(scores.items(), key=lambda x: x[1].probability)

        print(f"Best match: {best_slider[0]} (prob={best_slider[1].probability:.3f})")

        # Show all scores
        for name, output in scores.items():
            print(f"  {name}: {output.probability:.3f}")

    print("\n" + "="*60)
    print("INTEGRATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Replace example data with your actual data sources")
    print("2. Customize slider definitions for your use case")
    print("3. Add persistence for embeddings (Redis, PostgreSQL pgvector)")
    print("4. Implement API endpoint for real-time scoring")
    print("5. Set up monitoring for calibration drift")


if __name__ == "__main__":
    main()