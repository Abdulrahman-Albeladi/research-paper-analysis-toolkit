import numpy as np

from paper2_assignments.code_corpus import aggregate_chunk_scores, split_code_by_words


def test_code_chunking_and_aggregation():
    chunks = split_code_by_words("a b\nc d\ne f", max_words=2)
    assert len(chunks) == 3
    assert np.isclose(aggregate_chunk_scores([10, 90], [1, 3]), 70)
