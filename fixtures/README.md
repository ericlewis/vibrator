# Regression fixtures

Place anonymized or synthetic user interaction logs here to drive regression tests.

Expected JSON schema for `regression_logs.json`:

```json
{
  "slider_texts": {"slider_name": "Slider description"},
  "actions": [
    {
      "action_type": "write",
      "content": "User utterance",
      "weight": 1.0,
      "metadata": {"age_hours": 1.5}
    }
  ],
  "item": "Candidate content text",
  "context": {"segment": "optional segment string"},
  "expected_probabilities": {"slider_name": 0.5}
}
```

Replace the example values with regression targets derived from production data once available. The existing integration test will validate the probabilities within a small tolerance to catch embedding drift.
