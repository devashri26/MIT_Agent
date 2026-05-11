from backend.normalization.contamination_detector import detect_cross_domain_contamination
from backend.normalization.semantic_section_splitter import detect_mixed_topic, detect_paragraph_topic


def test_paragraph_topic_detects_eligibility() -> None:
    text = "Eligibility for MCA requires a relevant undergraduate degree and an entrance exam score."
    assert detect_paragraph_topic(text) == "eligibility"


def test_paragraph_topic_detects_placement() -> None:
    text = "Recruiters offered highest package of 25 LPA with average package 8 LPA across batches."
    assert detect_paragraph_topic(text) == "placements"


def test_paragraph_topic_detects_hostel() -> None:
    text = "The hostel accommodation provides residential rooms and mess for outstation students."
    assert detect_paragraph_topic(text) == "hostel"


def test_paragraph_topic_fallback_to_general() -> None:
    text = "The campus is located near scenic surroundings with abundant greenery and trees."
    assert detect_paragraph_topic(text) == "general"


def test_mixed_topic_eligibility_and_hostel() -> None:
    text = (
        "Eligibility for MCA requires a relevant undergraduate degree and entrance examination.\n\n"
        "The hostel accommodation offers residential rooms and mess service for outstation students.\n\n"
        "Eligibility for MTech requires a valid GATE score and strong academic record overall."
    )
    mixed, topics = detect_mixed_topic(text)
    assert mixed is True
    assert "eligibility" in topics
    assert "hostel" in topics


def test_single_topic_not_mixed() -> None:
    text = (
        "Eligibility for MCA requires a relevant undergraduate degree and entrance exam.\n\n"
        "Eligibility criteria also include minimum percentage requirements per regulations.\n\n"
        "Admission criteria for MTech require valid GATE scores submitted before deadlines."
    )
    mixed, _ = detect_mixed_topic(text)
    assert mixed is False


def test_short_chunk_not_mixed() -> None:
    mixed, _ = detect_mixed_topic("Short text.")
    assert mixed is False


def test_contamination_from_off_domain_components() -> None:
    is_contaminated, sources = detect_cross_domain_contamination(
        page_type="Research",
        component_types_in_chunk=["admissions_cta", "footer"],
        mixed_topic=False,
        dominant_topics=[],
    )
    assert is_contaminated is True
    assert "admissions_cta" in sources
    assert "footer" not in sources


def test_contamination_from_off_topic_mix() -> None:
    is_contaminated, sources = detect_cross_domain_contamination(
        page_type="Research",
        component_types_in_chunk=[],
        mixed_topic=True,
        dominant_topics=["hostel", "admissions"],
    )
    assert is_contaminated is True
    assert any(s.startswith("off_topic:") for s in sources)


def test_no_contamination_when_topics_match_page_type() -> None:
    is_contaminated, sources = detect_cross_domain_contamination(
        page_type="Admissions",
        component_types_in_chunk=[],
        mixed_topic=True,
        dominant_topics=["eligibility", "admissions"],
    )
    assert is_contaminated is False
    assert sources == []


def test_no_contamination_when_chunk_is_clean() -> None:
    is_contaminated, sources = detect_cross_domain_contamination(
        page_type="Research",
        component_types_in_chunk=[],
        mixed_topic=False,
        dominant_topics=[],
    )
    assert is_contaminated is False
    assert sources == []
