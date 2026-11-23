import numpy as np
from fedsight.matching import retrieve_and_rank


def make_track(track_id, source_key, embedding):
    return {
        "track_id": track_id,
        "source_key": source_key,
        "embedding": np.asarray(embedding, dtype=np.float32),
        "preview_rgb": np.zeros((12, 8, 3), dtype=np.uint8),
        "frame_count": 5,
        "avg_conf": 0.8,
    }


def test_matching_ranks_identical_embeddings_first():
    a = [make_track("A1", "A", [1, 0, 0]), make_track("A2", "A", [0, 1, 0])]
    b = [make_track("B1", "B", [0.99, 0.01, 0]), make_track("B2", "B", [0, 0.99, 0.01])]
    pairs, backend = retrieve_and_rank(a, b, top_k=1)
    assert {(p["track_a"]["track_id"], p["track_b"]["track_id"]) for p in pairs} == {("A1", "B1"), ("A2", "B2")}
    assert "cosine" in backend.lower() or "faiss" in backend.lower()
