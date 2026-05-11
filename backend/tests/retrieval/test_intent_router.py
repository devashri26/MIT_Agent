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
