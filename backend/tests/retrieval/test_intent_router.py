from backend.retrieval.intent_router import IntentRouter


def test_eligibility_query() -> None:
    route = IntentRouter().route("What is MCA eligibility?")
    assert route.intent == "eligibility_query"
    assert "Admissions" in route.allowed_page_types
    assert "Programs" in route.allowed_page_types
    assert "eligibility" in route.allowed_section_types


def test_fees_query() -> None:
    route = IntentRouter().route("What is the fee structure?")
    assert route.intent == "fees_query"
    assert "fees" in route.allowed_section_types


def test_placement_query() -> None:
    route = IntentRouter().route("Placement highest package")
    assert route.intent == "placement_query"
    assert route.allowed_page_types == ["Placements"]


def test_faculty_query() -> None:
    route = IntentRouter().route("Computer engineering HOD")
    assert route.intent == "faculty_query"
    assert "Faculty" in route.allowed_page_types


def test_hostel_query() -> None:
    route = IntentRouter().route("What are hostel facilities?")
    assert route.intent == "hostel_query"
    assert "Facilities" in route.allowed_page_types


def test_curriculum_query() -> None:
    route = IntentRouter().route("BTech AI curriculum")
    assert route.intent == "curriculum_query"
    assert "Curriculum" in route.allowed_page_types


def test_club_query() -> None:
    route = IntentRouter().route("IEEE student branch coordinator")
    assert route.intent == "club_query"


def test_event_query() -> None:
    route = IntentRouter().route("Cultural events fest")
    assert route.intent == "event_query"


def test_research_query() -> None:
    route = IntentRouter().route("Research publications computer engineering")
    assert route.intent == "research_query"


def test_general_query_fallback() -> None:
    route = IntentRouter().route("MITAOE Alandi Pune")
    assert route.intent == "general_query"
    assert "Admissions" in route.allowed_page_types


def test_dean_routes_to_faculty_with_wider_page_types() -> None:
    """Regression: 'dean of internship and placement cell' was routing to placement_query
    which restricts to page_type=Placements. The leadership info actually lives on
    General/Programs pages, so a dean/director query must now route to faculty_query
    with broader allowed page_types."""
    route = IntentRouter().route("who is the dean of internship and placement cell")
    assert route.intent == "faculty_query"
    assert "Faculty" in route.allowed_page_types
    assert "Programs" in route.allowed_page_types
    assert "General" in route.allowed_page_types


def test_director_also_routes_to_faculty() -> None:
    route = IntentRouter().route("who is director of mitaoe")
    assert route.intent == "faculty_query"


def test_placement_query_still_works_without_leadership_words() -> None:
    """Regression check: a plain placement query (no dean/director) must still hit
    placement_query, not faculty_query."""
    route = IntentRouter().route("mechanical placement statistics")
    assert route.intent == "placement_query"
    assert route.allowed_page_types == ["Placements"]


def test_general_query_includes_blog_for_explanatory_content() -> None:
    """Regression: 'what do you know about spot round admission' is generic and
    explanatory content lives in Blog page_types (MHT-CET counselling guides).
    general_query must include Blog so this content surfaces."""
    route = IntentRouter().route("MITAOE Alandi Pune")
    assert "Blog" in route.allowed_page_types
