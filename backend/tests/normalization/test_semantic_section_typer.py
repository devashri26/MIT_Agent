from backend.normalization.semantic_section_typer import SemanticSectionTyper


def test_heading_detects_eligibility() -> None:
    typer = SemanticSectionTyper()
    assert typer.type_section("Eligibility Criteria", [], "Some text.", "Admissions") == "eligibility"


def test_heading_detects_fees() -> None:
    typer = SemanticSectionTyper()
    assert typer.type_section("Fee Structure", [], "", "Admissions") == "fees"


def test_heading_detects_hostel() -> None:
    typer = SemanticSectionTyper()
    assert typer.type_section("Hostel Accommodation", [], "", "Facilities") == "hostel"


def test_faq_marker_in_text() -> None:
    typer = SemanticSectionTyper()
    text = "Q1: What is the fee?\nA: 1 lakh.\nQ2: Hostel charges?\nA: Extra."
    assert typer.type_section("Information", [], text, "General") == "faq"


def test_statistics_marker_in_text() -> None:
    typer = SemanticSectionTyper()
    text = "Placement summary: 95% placed with average package 12 LPA. Highest 45 LPA."
    assert typer.type_section("Summary", [], text, "Placements") == "statistics"


def test_page_type_fallback_when_no_signals() -> None:
    typer = SemanticSectionTyper()
    assert typer.type_section("", [], "Welcome.", "Faculty") == "faculty"
    assert typer.type_section("", [], "Welcome.", "Research") == "research"
    assert typer.type_section("", [], "Welcome.", "Blog") == "general"
