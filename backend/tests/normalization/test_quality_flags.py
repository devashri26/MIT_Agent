from backend.normalization.quality_flags import compute_quality_flags


def test_low_content_flag_for_short_chunks() -> None:
    flags = compute_quality_flags("Short.", token_count=10, page_type_confidence=0.95, page_type="General", quality_score=1.0)
    assert "low_content" in flags


def test_thin_content_for_medium_low_quality() -> None:
    flags = compute_quality_flags(
        "Some content here.",
        token_count=100,
        page_type_confidence=0.95,
        page_type="General",
        quality_score=0.3,
    )
    assert "thin_content" in flags


def test_weak_classification_when_confidence_low() -> None:
    flags = compute_quality_flags("ok", token_count=500, page_type_confidence=0.4, page_type="General", quality_score=1.0)
    assert "weak_classification" in flags


def test_event_page_flag_for_events_page_type() -> None:
    flags = compute_quality_flags("ok", token_count=500, page_type_confidence=0.9, page_type="Events", quality_score=1.0)
    assert "event_page" in flags


def test_boilerplate_heavy_flag() -> None:
    text = "Copyright 2024 MITAOE. All rights reserved. Privacy policy applies. Terms of use govern."
    flags = compute_quality_flags(text, token_count=200, page_type_confidence=0.9, page_type="General", quality_score=1.0)
    assert "boilerplate_heavy" in flags


def test_no_flags_for_good_chunk() -> None:
    flags = compute_quality_flags(
        "MCA admissions require a relevant degree. The minimum qualification is 50 percent aggregate.",
        token_count=500,
        page_type_confidence=0.95,
        page_type="Admissions",
        quality_score=0.9,
    )
    assert flags == []
