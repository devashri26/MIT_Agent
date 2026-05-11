import math

from backend.reranking.score_calibrator import calibrate, combine_relevance, sigmoid


def test_sigmoid_midpoint() -> None:
    assert abs(sigmoid(0.0) - 0.5) < 1e-9


def test_sigmoid_monotonic() -> None:
    values = [-5.0, -1.0, 0.0, 1.0, 5.0]
    outputs = [sigmoid(v) for v in values]
    assert outputs == sorted(outputs)


def test_sigmoid_extremes_stable() -> None:
    assert math.isfinite(sigmoid(1000.0))
    assert math.isfinite(sigmoid(-1000.0))
    assert sigmoid(1000.0) > 0.99
    assert sigmoid(-1000.0) < 0.01


def test_calibrate_rounds_to_four() -> None:
    assert calibrate(2.5) == round(sigmoid(2.5), 4)


def test_combine_relevance_default_weights() -> None:
    result = combine_relevance(rerank_calibrated=1.0, answerability=1.0)
    assert abs(result - 1.0) < 1e-9
    result = combine_relevance(rerank_calibrated=0.0, answerability=0.0)
    assert result == 0.0


def test_combine_relevance_respects_weights() -> None:
    rerank_only = combine_relevance(1.0, 0.0, rerank_weight=0.8, answerability_weight=0.2)
    assert abs(rerank_only - 0.8) < 1e-9
    answer_only = combine_relevance(0.0, 1.0, rerank_weight=0.8, answerability_weight=0.2)
    assert abs(answer_only - 0.2) < 1e-9
