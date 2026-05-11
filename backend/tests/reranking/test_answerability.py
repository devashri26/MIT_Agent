from backend.reranking.answerability import compute_answerability_score


def test_eligibility_text_scores_high() -> None:
    text = (
        "Eligibility for MCA: minimum 50% in graduation. Fee structure ₹120000 per year. "
        "Semester credits and core courses are listed in the curriculum."
    )
    score = compute_answerability_score(text, token_count=200)
    assert score >= 0.6


def test_marketing_text_scores_low() -> None:
    text = (
        "Welcome to one of the leading institutions in Pune. World-class infrastructure "
        "and nurturing journey. Apply now! Click here. Read more. Download the brochure."
    )
    score = compute_answerability_score(text, token_count=100)
    assert score <= 0.45


def test_faq_marker_boosts() -> None:
    no_faq = compute_answerability_score("Some general info about the campus.", token_count=200)
    faq = compute_answerability_score("Q1: What is the eligibility? A: 50% in graduation.", token_count=200)
    assert faq > no_faq


def test_empty_text_zero() -> None:
    assert compute_answerability_score("", token_count=0) == 0.0


def test_stats_present_boosts() -> None:
    text = "Placement statistics 2024: 95% placed with average package 12 LPA and highest 45 LPA."
    score = compute_answerability_score(text, token_count=200)
    assert score >= 0.55
