import numpy as np


def dense_ranker(article_emb: np.ndarray, article_ids: np.ndarray,
                 query_emb: np.ndarray, top_k: int = 10) -> list[list[int]]:
    sim = article_emb @ query_emb.T
    rankings = []
    for i in range(query_emb.shape[0]):
        scores = sim[:, i]
        rankings.append(article_ids[np.argsort(scores)[::-1][:top_k]].tolist())
    return rankings
