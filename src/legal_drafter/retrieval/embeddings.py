from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Sequence

import numpy as np

DENSE_DIM = 1024
BGE_MODEL_ID = "BAAI/bge-m3"
RERANKER_ID = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _bge_model():
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(BGE_MODEL_ID, use_fp16=False)


@lru_cache(maxsize=1)
def _reranker():
    # sentence-transformers CrossEncoder supports transformers>=5; FlagEmbedding's
    # FlagReranker uses tokenizer APIs removed in transformers 5, so we prefer this.
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(RERANKER_ID)
    except ImportError:  # pragma: no cover - fallback for older envs
        from FlagEmbedding import FlagReranker

        return FlagReranker(RERANKER_ID, use_fp16=False)


def encode(texts: Sequence[str]) -> tuple[List[List[float]], List[dict]]:
    """Encode texts into (dense vectors, sparse vectors).

    sparse vectors are returned as {"indices": [int], "values": [float]} ready
    for Qdrant's sparse vector field.
    """
    model = _bge_model()
    out = model.encode(
        list(texts),
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=8192,
    )
    dense = out["dense_vecs"]
    if hasattr(dense, "tolist"):
        dense = dense.tolist()
    dense = [list(map(float, v)) for v in dense]
    sparse = [lexical_weights_to_sparse(w) for w in out["lexical_weights"]]
    return dense, sparse


def lexical_weights_to_sparse(lexical_weights: dict) -> dict:
    """Convert BGE-M3 lexical_weights ({token_id: weight}) to a sorted sparse vector."""
    if not lexical_weights:
        return {"indices": [], "values": []}
    indices = [int(t) for t in lexical_weights.keys()]
    values = [float(w) for w in lexical_weights.values()]
    order = np.argsort(indices)
    indices = [indices[i] for i in order]
    values = [values[i] for i in order]
    return {"indices": indices, "values": values}


def rerank_scores(query: str, docs: Sequence[str], *, normalize: bool = True) -> List[float]:
    """Return a relevance score for each (query, doc) pair; higher = more relevant."""
    if not docs:
        return []
    model = _reranker()
    pairs = [[query, d] for d in docs]
    scores = model.predict(pairs)
    scores = [float(s) for s in scores]
    if normalize:
        scores = [1.0 / (1.0 + math.exp(-s)) for s in scores]
    return scores


def rerank(query: str, docs: Sequence[str], top_k: int | None = None) -> List[int]:
    scores = rerank_scores(query, docs, normalize=True)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order if top_k is None else order[:top_k]
