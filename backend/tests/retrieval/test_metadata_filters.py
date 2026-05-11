from backend.retrieval.metadata_filters import exclude_reusable_components, filter_by_page_types


def test_filter_keeps_allowed_types() -> None:
    chunks = [
        {"page_type": "Admissions"},
        {"page_type": "Blog"},
        {"page_type": "Programs"},
    ]
    indices = filter_by_page_types(chunks, ["Admissions", "Programs"])
    assert indices == [0, 2]


def test_empty_allowed_returns_all() -> None:
    chunks = [{"page_type": "Blog"}, {"page_type": "General"}]
    assert filter_by_page_types(chunks, None) == [0, 1]
    assert filter_by_page_types(chunks, []) == [0, 1]


def test_no_matches_returns_empty_list() -> None:
    chunks = [{"page_type": "Blog"}, {"page_type": "Events"}]
    assert filter_by_page_types(chunks, ["Admissions"]) == []


def test_exclude_reusable_components_drops_flagged() -> None:
    chunks = [
        {"page_type": "Admissions", "is_reusable_component": False},
        {"page_type": "Admissions", "is_reusable_component": True},
        {"page_type": "Admissions", "is_reusable_component": False},
    ]
    kept, excluded = exclude_reusable_components(chunks, [0, 1, 2])
    assert kept == [0, 2]
    assert excluded == 1


def test_exclude_reusable_components_keeps_all_when_none_flagged() -> None:
    chunks = [{"is_reusable_component": False}, {"is_reusable_component": False}]
    kept, excluded = exclude_reusable_components(chunks, [0, 1])
    assert kept == [0, 1]
    assert excluded == 0
