import numpy as np
import pytest

from vibrator import InstructionalEncoder, SliderScorer
from vibrator.synthetic import synthetic_actions, synthetic_item_corpus, synthetic_slider_texts


@pytest.fixture(scope="session")
def encoder() -> InstructionalEncoder:
    return InstructionalEncoder()


def test_instructional_encoder_normalization(encoder: InstructionalEncoder):
    slider_texts = synthetic_slider_texts(limit=2)
    embeddings = encoder.encode_items(slider_texts.values())
    assert embeddings.shape[0] == 2
    # Layer normalization keeps mean ~0, std ~1 before L2
    assert np.allclose(embeddings.mean(axis=1), 0.0, atol=1e-3)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_pipeline_with_real_encoder(encoder: InstructionalEncoder):
    slider_texts = synthetic_slider_texts(limit=3)
    slider_vectors = dict(zip(slider_texts.keys(), encoder.encode_items(slider_texts.values())))

    actions = synthetic_actions(count=4, seed=21)
    item = synthetic_item_corpus(limit=1)[0]

    scorer = SliderScorer(encoder, slider_vectors)
    outputs = scorer.score(actions, item_text=item, context={"segment": "integration"})

    assert set(outputs.keys()) == set(slider_vectors.keys())
    for output in outputs.values():
        assert 0.0 < output.probability < 1.0
        assert set(output.features.keys()) == {"user_to_slider", "item_to_slider", "user_item_alignment", "recency"}
        assert abs(sum(output.weights.values()) - 1.0) < 1e-6
