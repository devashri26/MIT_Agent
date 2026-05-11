from backend.retrieval.query_expansion import expand_query


def test_expansion_includes_originals() -> None:
    original, expanded = expand_query("MCA eligibility")
    assert original == ["mca", "eligibility"]
    assert "mca" in expanded
    assert "eligibility" in expanded


def test_expansion_adds_synonyms() -> None:
    _, expanded = expand_query("eligibility")
    assert "requirements" in expanded
    assert "criteria" in expanded


def test_expansion_no_synonyms_for_unknown_term() -> None:
    original, expanded = expand_query("foobar")
    assert original == ["foobar"]
    assert expanded == ["foobar"]


def test_expansion_unique_tokens() -> None:
    _, expanded = expand_query("hostel hostel")
    assert expanded.count("hostel") == 1


def test_expansion_handles_empty() -> None:
    original, expanded = expand_query("")
    assert original == []
    assert expanded == []
