"""
Agent 1 — Retrieval.

Finds the k most similar past tasks for a new description, to be used as
few-shot examples by the inference agent. Embeddings rather than keyword
overlap, because the evaluation set contains unseen and deliberately vague
descriptions that share almost no vocabulary with the repository.

Vectors are normalised at encoding time, so cosine similarity is just the dot
product and the whole search is one matrix multiplication. 4,500 x 384 floats
is 6.6 MB: numpy is enough, a vector database would add a dependency for a
speed-up that is invisible at this scale.
"""

import numpy as np
import pandas as pd

import config

_encoder = None


def get_encoder():
    """Loaded once, on first use. The import is inside the function because
    sentence_transformers pulls in PyTorch and takes ~7 s: anything that only
    reads the cached index should not pay that."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(config.EMBEDDING_MODEL_ID)
    return _encoder


def get_index(historical_df: pd.DataFrame, rebuild: bool = False) -> np.ndarray:
    """The embedding matrix, built on first use and cached to disk.

    Row i of the matrix is row i of the CSV — the link is positional, so a
    matrix whose length no longer matches the repository is silently wrong.
    Rebuilding automatically in that case removes the failure mode entirely.
    """
    if config.EMBEDDINGS_FILE.exists() and not rebuild:
        embeddings = np.load(config.EMBEDDINGS_FILE)
        if len(embeddings) == len(historical_df):
            return embeddings
        print("Embedding index is stale, rebuilding...")

    print(f"Encoding {len(historical_df)} descriptions (two or three minutes)...")
    embeddings = get_encoder().encode(
        historical_df["description"].tolist(),
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    ).astype(np.float32)

    config.EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_FILE, embeddings)
    return embeddings


def retrieve(descriptions, historical_df: pd.DataFrame,
             embeddings: np.ndarray, k: int = None):
    """The k most similar repository rows, with a 'similarity' column.
    """
    k = k or config.RETRIEVAL_TOP_K
    single = isinstance(descriptions, str)
    queries = get_encoder().encode(
        [descriptions] if single else list(descriptions),
        normalize_embeddings=True,
        batch_size=64,
    )

    results = []
    for similarities in queries @ embeddings.T:   # every query vs every task
        top = np.argsort(-similarities)[:k]
        frame = historical_df.iloc[top].copy()    # copy: never mutate the repo
        frame["similarity"] = similarities[top]
        results.append(frame)

    return results[0] if single else results