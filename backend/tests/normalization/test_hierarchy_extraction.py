from backend.normalization.heading_classifier import is_generic_heading
from backend.normalization.hierarchy_extractor import (
    clean_document_title,
    humanize_page_type,
    humanize_section_type,
)
from backend.normalization.section_normalizer import (
    needs_hierarchy_repair,
    normalize_section_path,
)


def test_is_generic_heading() -> None:
    assert is_generic_heading("Overview") is True
    assert is_generic_heading("Introduction") is True
    assert is_generic_heading("General") is True
    assert is_generic_heading("") is True
    assert is_generic_heading("home") is True
    assert is_generic_heading("Eligibility Criteria") is False
    assert is_generic_heading("MCA Admissions") is False


def test_humanize_section_type() -> None:
    assert humanize_section_type("eligibility") == "Eligibility"
    assert humanize_section_type("faq") == "FAQ"
    assert humanize_section_type("overview") == ""
    assert humanize_section_type("general") == ""


def test_humanize_page_type() -> None:
    assert humanize_page_type("Admissions") == "Admissions"
    assert humanize_page_type("Club") == "Student Clubs"
    assert humanize_page_type("General") == ""


def test_clean_document_title_strips_brand_suffix() -> None:
    assert clean_document_title("MCA Admissions | MIT Academy of Engineering") == "MCA Admissions"


def test_clean_document_title_strips_brand_prefix() -> None:
    assert clean_document_title("MITAOE | Notice Board") == "Notice Board"


def test_clean_document_title_no_separator() -> None:
    assert clean_document_title("Computer Engineering Course Structure") == "Computer Engineering Course Structure"


def test_normalize_section_path_builds_full_hierarchy() -> None:
    path = normalize_section_path(
        page_type="Admissions",
        section_type="eligibility",
        document_title="MCA Admissions | MIT Academy of Engineering",
    )
    assert path == ["Admissions", "Eligibility", "MCA Admissions"]


def test_normalize_section_path_drops_generics() -> None:
    path = normalize_section_path(
        page_type="General",
        section_type="overview",
        document_title="MITAOE | Home",
    )
    assert path == []


def test_normalize_section_path_dedupes_case_insensitive() -> None:
    path = normalize_section_path(
        page_type="Admissions",
        section_type="admissions",
        document_title="Admissions",
    )
    assert path == ["Admissions"]


def test_normalize_section_path_includes_metadata_headings() -> None:
    path = normalize_section_path(
        page_type="Admissions",
        section_type="eligibility",
        document_title="MCA Admissions",
        metadata_headings=["Overview", "Programs Offered"],
    )
    assert "Programs Offered" in path
    assert "Overview" not in path


def test_needs_hierarchy_repair() -> None:
    assert needs_hierarchy_repair("Overview") is True
    assert needs_hierarchy_repair("Eligibility Criteria") is False
