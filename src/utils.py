import numpy as np


def average_precision_at_k(ground_truth: set, ranked: list, k: int) -> float:
    if not ground_truth:
        return 0.0

    ranked = ranked[:k]
    relevant = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(ranked):
        if doc_id in ground_truth:
            relevant += 1
            sum_precision += relevant / (i + 1)

    return sum_precision / min(len(ground_truth), k)


def mean_average_precision_at_k(
    ground_truths: list[set[int]], rankings: list[list[int]], k: int
) -> float:
    scores = [
        average_precision_at_k(gt, ranking, k)
        for gt, ranking in zip(ground_truths, rankings)
    ]
    return float(np.mean(scores))
