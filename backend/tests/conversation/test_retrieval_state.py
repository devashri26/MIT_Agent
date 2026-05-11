from backend.conversation.retrieval_state import extract_entities


def test_extracts_program_mca() -> None:
    assert extract_entities("What is MCA eligibility?")["program"] == "MCA"


def test_extracts_program_btech() -> None:
    assert extract_entities("BTech curriculum semester 1")["program"] == "BTech"


def test_extracts_department_mechanical() -> None:
    found = extract_entities("Mechanical engineering placements")
    assert found["department"] == "Mechanical Engineering"


def test_extracts_both_program_and_department() -> None:
    found = extract_entities("BTech Computer Engineering fees")
    assert found["program"] == "BTech"
    assert found["department"] == "Computer Engineering"


def test_returns_empty_for_no_entities() -> None:
    assert extract_entities("Tell me about the campus") == {}
