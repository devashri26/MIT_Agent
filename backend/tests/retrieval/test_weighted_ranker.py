import pytest

from backend.retrieval.weighted_ranker import combine, normalize_scores, rank_candidates, section_match_bonus


def test_normalize_scores_max_normalizes() -> None:
    assert normalize_scores([2.0, 1.0, 0.5]) == [1.0, 0.5, 0.25]


def test_normalize_scores_zero_max() -> None:
    assert normalize_scores([0.0, 0.0]) == [0.0, 0.0]


def test_section_match_bonus_present() -> None:
    assert section_match_bonus("eligibility", ["eligibility", "admissions"]) == 1.0


def test_section_match_bonus_absent() -> None:
    assert section_match_bonus("overview", ["eligibility"]) == 0.0
    assert section_match_bonus(None, ["eligibility"]) == 0.0
    assert section_match_bonus("overview", []) == 0.0


def test_combine_weighted_formula() -> None:
    assert combine(bm25_normalized=1.0, retrieval_priority=1.0, section_match=1.0) == pytest.approx(1.0)
    assert combine(bm25_normalized=0.0, retrieval_priority=0.0, section_match=0.0) == 0.0
    assert combine(bm25_normalized=1.0, retrieval_priority=0.0, section_match=0.0) == pytest.approx(0.7)
    assert combine(bm25_normalized=0.0, retrieval_priority=1.0, section_match=0.0) == pytest.approx(0.2)
    assert combine(bm25_normalized=0.0, retrieval_priority=0.0, section_match=1.0) == pytest.approx(0.1)


def test_rank_candidates_orders_by_final() -> None:
    chunks = [
        {"retrieval_priority": 1.0, "section_type": "eligibility"},
        {"retrieval_priority": 0.4, "section_type": "overview"},
        {"retrieval_priority": 0.9, "section_type": "admissions"},
    ]
    candidates = [(0, 5.0), (1, 10.0), (2, 8.0)]
    ranked = rank_candidates(candidates, chunks, ["eligibility", "admissions"])
    ranks = [idx for idx, _ in ranked]
    # Chunk 0 has section match + high priority but lower bm25, chunk 1 has highest bm25 but no section/priority.
    # Combined: 0 → 0.7*0.5 + 0.2*1.0 + 0.1*1.0 = 0.65; 1 → 0.7*1.0 + 0.2*0.4 + 0.1*0 = 0.78; 2 → 0.7*0.8 + 0.2*0.9 + 0.1*1.0 = 0.84
    assert ranks[0] == 2
    assert ranks[1] == 1
    assert ranks[2] == 0


def test_rank_candidates_empty() -> None:
    assert rank_candidates([], [], []) == []
