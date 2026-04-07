"""Unit tests for BM25, RRF, and reranker — no external services."""
import pytest

from knowledge_engine.retrieval.bm25_index import BM25Index
from knowledge_engine.retrieval.rrf_fusion import reciprocal_rank_fusion


CORPUS = [
    {"chunk_id": "aapl_mda_0001", "text": "Apple revenue was $383 billion in fiscal 2023."},
    {"chunk_id": "aapl_mda_0002", "text": "iPhone sales represented 52% of total revenue."},
    {"chunk_id": "msft_mda_0001", "text": "Microsoft Azure cloud revenue grew 28% year-over-year."},
    {"chunk_id": "msft_mda_0002", "text": "Microsoft total revenue was $211.9 billion."},
    {"chunk_id": "amzn_mda_0001", "text": "Amazon AWS revenue reached $90.8 billion in 2023."},
]


class TestBM25Index:
    @pytest.fixture
    def index(self) -> BM25Index:
        idx = BM25Index()
        idx.build(CORPUS)
        return idx

    def test_search_returns_results(self, index: BM25Index) -> None:
        results = index.search("Apple revenue 2023", top_k=3)
        assert len(results) == 3

    def test_search_apple_query_ranks_apple_first(self, index: BM25Index) -> None:
        results = index.search("Apple iPhone revenue fiscal 2023", top_k=3)
        chunk_ids = [r["chunk_id"] for r in results]
        assert any("aapl" in cid for cid in chunk_ids[:2])

    def test_search_azure_query_ranks_microsoft(self, index: BM25Index) -> None:
        results = index.search("Microsoft Azure cloud growth", top_k=3)
        assert results[0]["chunk_id"] == "msft_mda_0001"

    def test_result_has_required_fields(self, index: BM25Index) -> None:
        results = index.search("revenue", top_k=1)
        assert "chunk_id" in results[0]
        assert "text" in results[0]
        assert "score" in results[0]

    def test_search_before_build_raises(self) -> None:
        idx = BM25Index()
        with pytest.raises(RuntimeError, match="build\\(\\) must be called"):
            idx.search("query")

    def test_top_k_limits_results(self, index: BM25Index) -> None:
        results = index.search("revenue", top_k=2)
        assert len(results) == 2


class TestRRFFusion:
    def test_combines_two_lists(self) -> None:
        list1 = [
            {"chunk_id": "a", "text": "text a"},
            {"chunk_id": "b", "text": "text b"},
        ]
        list2 = [
            {"chunk_id": "b", "text": "text b"},
            {"chunk_id": "c", "text": "text c"},
        ]
        result = reciprocal_rank_fusion([list1, list2])
        # "b" appears in both — should score highest
        assert result[0]["chunk_id"] == "b"

    def test_rrf_scores_decrease_monotonically(self) -> None:
        list1 = [{"chunk_id": str(i), "text": f"chunk {i}"} for i in range(5)]
        result = reciprocal_rank_fusion([list1])
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_output(self) -> None:
        list1 = [{"chunk_id": str(i), "text": f"chunk {i}"} for i in range(10)]
        result = reciprocal_rank_fusion([list1], top_n=3)
        assert len(result) == 3

    def test_empty_input_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_deduplicates_across_lists(self) -> None:
        same_item = {"chunk_id": "x", "text": "shared"}
        list1 = [same_item]
        list2 = [same_item]
        result = reciprocal_rank_fusion([list1, list2])
        chunk_ids = [r["chunk_id"] for r in result]
        assert chunk_ids.count("x") == 1
