from backend.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_rrf_top1_appears_first_when_in_both() -> None:
    fused = reciprocal_rank_fusion(["a", "b", "c"], ["a", "x", "y"])
    assert fused[0].chunk_id == "a"
    assert "bm25" in fused[0].sources and "dense" in fused[0].sources


def test_rrf_includes_all_unique_chunks() -> None:
    fused = reciprocal_rank_fusion(["a", "b"], ["c", "d"])
    assert {f.chunk_id for f in fused} == {"a", "b", "c", "d"}


def test_rrf_score_is_sum_of_reciprocals() -> None:
    fused = reciprocal_rank_fusion(["a"], ["a"])
    expected = 1.0 / (DEFAULT_RRF_K + 1) + 1.0 / (DEFAULT_RRF_K + 1)
    assert abs(fused[0].fusion_score - expected) < 1e-6
    assert fused[0].bm25_rank == 1
    assert fused[0].dense_rank == 1


def test_rrf_records_per_retriever_rank() -> None:
    fused = reciprocal_rank_fusion(["a", "b", "c"], ["c", "a"])
    by_id = {f.chunk_id: f for f in fused}
    assert by_id["a"].bm25_rank == 1
    assert by_id["a"].dense_rank == 2
    assert by_id["c"].bm25_rank == 3
    assert by_id["c"].dense_rank == 1
    assert by_id["b"].dense_rank == 0


def test_rrf_chunk_only_in_dense_has_zero_bm25_rank() -> None:
    fused = reciprocal_rank_fusion(["a"], ["b"])
    by_id = {f.chunk_id: f for f in fused}
    assert by_id["b"].bm25_rank == 0
    assert by_id["b"].dense_rank == 1
    assert by_id["b"].sources == ["dense"]


def test_rrf_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_orders_by_descending_score() -> None:
    fused = reciprocal_rank_fusion(["a", "b", "c", "d"], ["c", "b", "a"])
    scores = [f.fusion_score for f in fused]
    assert scores == sorted(scores, reverse=True)
