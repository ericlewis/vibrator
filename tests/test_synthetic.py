from vibrator.synthetic import (
    ACTION_TYPES,
    synthetic_actions,
    synthetic_item_corpus,
    synthetic_slider_texts,
)


def test_synthetic_slider_texts_limit():
    limited = synthetic_slider_texts(limit=2)
    assert len(limited) == 2
    assert list(limited.keys()) == ["professional", "personal"]


def test_synthetic_item_corpus_limit():
    items = synthetic_item_corpus(limit=3)
    assert len(items) == 3
    assert items[0].startswith("Async")


def test_synthetic_actions_reproducible():
    actions_a = synthetic_actions(count=3, seed=7)
    actions_b = synthetic_actions(count=3, seed=7)
    assert [a.action_type for a in actions_a] == [b.action_type for b in actions_b]
    assert [a.weight for a in actions_a] == [b.weight for b in actions_b]
    assert all(action.action_type in ACTION_TYPES for action in actions_a)
    assert all("age_hours" in action.metadata for action in actions_a)
