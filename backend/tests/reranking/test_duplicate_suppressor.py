from backend.reranking.duplicate_suppressor import suppress_semantic_duplicates


def test_exact_duplicate_detected() -> None:
    cands = [
        {"chunk_id": "a", "text": "MCA eligibility minimum 50 percent in graduation."},
        {"chunk_id": "b", "text": "MCA eligibility minimum 50 percent in graduation."},
    ]
    decisions = suppress_semantic_duplicates(cands)
    assert decisions == [None, "a"]


def test_near_duplicate_detected() -> None:
    cands = [
        {"chunk_id": "a", "text": "MCA admissions require a relevant degree and entrance score."},
        {"chunk_id": "b", "text": "Relevant degree and entrance score required for MCA admissions."},
    ]
    decisions = suppress_semantic_duplicates(cands, similarity_threshold=80)
    assert decisions[0] is None
    assert decisions[1] == "a"


def test_unrelated_chunks_not_duplicates() -> None:
    cands = [
        {"chunk_id": "a", "text": "MCA eligibility minimum qualification."},
        {"chunk_id": "b", "text": "Hostel accommodation includes mess and rooms."},
    ]
    decisions = suppress_semantic_duplicates(cands)
    assert decisions == [None, None]


def test_first_occurrence_wins() -> None:
    cands = [
        {"chunk_id": "a", "text": "MCA admissions eligibility minimum criteria."},
        {"chunk_id": "b", "text": "MCA admissions eligibility minimum criteria."},
        {"chunk_id": "c", "text": "MCA admissions eligibility minimum criteria."},
    ]
    decisions = suppress_semantic_duplicates(cands)
    assert decisions == [None, "a", "a"]
