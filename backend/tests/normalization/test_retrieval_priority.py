from backend.normalization.retrieval_priority import compute_retrieval_priority


def test_admissions_outranks_notices() -> None:
    admissions = compute_retrieval_priority("Admissions", [], 0.95, 1.0)
    notices = compute_retrieval_priority("Notices", [], 0.95, 1.0)
    assert admissions > notices


def test_programs_top_tier() -> None:
    score = compute_retrieval_priority("Programs", [], 0.95, 1.0)
    assert score >= 0.85


def test_duplicate_flag_penalizes() -> None:
    clean = compute_retrieval_priority("Admissions", [], 0.95, 1.0)
    dup = compute_retrieval_priority("Admissions", ["duplicate"], 0.95, 1.0)
    assert dup < clean - 0.3


def test_low_confidence_dampens_score() -> None:
    high = compute_retrieval_priority("Admissions", [], 0.95, 1.0)
    low = compute_retrieval_priority("Admissions", [], 0.3, 1.0)
    assert low < high


def test_score_clamped_to_unit_interval() -> None:
    score = compute_retrieval_priority("Admissions", ["duplicate", "non_canonical", "low_content"], 0.4, 0.1)
    assert 0.0 <= score <= 1.0


def test_unknown_page_type_default() -> None:
    score = compute_retrieval_priority("Unknown", [], 0.95, 1.0)
    assert score > 0.0
